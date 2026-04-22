{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      lib = nixpkgs.lib;
      system = "x86_64-linux";

      gpuTargets = [
        "gfx90a"
        "gfx942"
        "gfx950"
        "all"
      ];

      sharedTools = pkgs: [
        pkgs.ruff
        # xkcd-font enables hand-drawn styling via matplotlib.pyplot.xkcd()
        pkgs.xkcd-font
      ];

      commonPythonPkgs = ps: [
        ps.numpy
        ps.transformers
        ps.datasets
        ps.safetensors
        ps.packaging
        ps.pyyaml
        ps.tqdm
        ps.matplotlib
        ps.zstandard
      ];

      gpuPythonPkgs =
        ps:
        (commonPythonPkgs ps)
        ++ [
          ps.torch
          ps.scipy
          ps.scikit-learn
          ps.accelerate
          ps.wandb
          ps.bitsandbytes
          ps.peft
          ps.einops
        ];

      mkPkgs =
        gpuTarget:
        import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            rocmSupport = true;
          };
          overlays = [
            (
              final: prev:
              let
                rocmScope = prev.rocmPackages.overrideScope (
                  self: super: {
                    clr = super.clr.override {
                      localGpuTargets = [ gpuTarget ];
                    };
                  }
                );
              in
              lib.optionalAttrs (gpuTarget != "all") {
                rocmPackages = rocmScope;
              }
            )
          ];
        };

      mkShell =
        gpuTarget:
        let
          pkgs = mkPkgs gpuTarget;
        in
        pkgs.mkShell {
          packages =
            (sharedTools pkgs)
            ++ (with pkgs; [
              # ROCm
              rocmPackages.rocm-smi
              rocmPackages.clr
              rocmPackages.rocminfo

              # Profiling / tracing
              rocmPackages.rocprofiler-sdk
              rocmPackages.rocprofiler
              rocmPackages.roctracer
              rocmPackages.rocr-debug-agent
              rocmPackages.rocprof-trace-decoder
              rocmPackages.aqlprofile
              rocmPackages.rdc

              (python3.withPackages gpuPythonPkgs)
            ]);

          env = {
            ROCM_PATH = "${pkgs.rocmPackages.clr}";
            GPU_ARCH = gpuTarget;
            GPU_ARCH_LIST = gpuTarget;
            GPU_ARCHS = gpuTarget;
          };
        };
      # Lightweight shell for dev on machines without GPUs
      defaultPkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      # CUDA shell for NVIDIA boxes — mirrors the ROCm shells' Python env
      # but builds torch with CUDA support. Useful for running eval locally
      # on checkpoints pulled back from the ROCm training host.
      cudaPkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          # RTX 4090 (Ada Lovelace, sm_89)
          cudaCapabilities = [ "8.9" ];
          cudaForwardCompat = false;
        };
      };

      # Paper (nl_probes/utils/common.py:26) loads Qwen3 with
      # attn_implementation="flash_attention_2". Upstream setup.py defaults
      # to building 80;90;100;120 arches (selected via FLASH_ATTN_CUDA_ARCHS
      # env, setup.py:69); nixpkgs doesn't set it and TORCH_CUDA_ARCH_LIST
      # isn't honored here. Also needs gcc14 since CUDA 12.9 rejects g++15.
      flashAttn =
        (cudaPkgs.python3Packages.flash-attn.override {
          buildPythonPackage = cudaPkgs.python3Packages.buildPythonPackage.override {
            stdenv = cudaPkgs.overrideCC cudaPkgs.stdenv cudaPkgs.gcc14;
          };
        }).overridePythonAttrs
          (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ cudaPkgs.gcc14 ];
            env = (old.env or { }) // {
              FLASH_ATTN_CUDA_ARCHS = "80"; # sm_80; Ada (sm_89) is binary-compatible
            };
          });

      cudaShell = cudaPkgs.mkShell {
        packages =
          (sharedTools cudaPkgs)
          ++ (with cudaPkgs; [
            cudaPackages.cuda_nvcc
            cudaPackages.cudatoolkit

            (python3.withPackages (ps: (gpuPythonPkgs ps) ++ [ flashAttn ]))
          ]);

        env = {
          LD_LIBRARY_PATH = "/run/opengl-driver/lib";
          CUDA_PATH = "${cudaPkgs.cudaPackages.cudatoolkit}";
        };
      };
    in
    {
      devShells.${system} = lib.genAttrs gpuTargets mkShell // {
        default = defaultPkgs.mkShell {
          packages = (sharedTools defaultPkgs) ++ [
            (defaultPkgs.python3.withPackages commonPythonPkgs)
          ];
        };
        cuda = cudaShell;
      };

      packages.${system} = lib.listToAttrs (
        map (gpuTarget: {
          name = "${gpuTarget}-rocprofiler-sdk";
          value = (mkPkgs gpuTarget).rocmPackages.rocprofiler-sdk;
        }) gpuTargets
      );
    };
}
