{
  description = "DevShell for stochastic variance project.";
  
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  
  outputs = { self, nixpkgs }: 
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        # Add your python packages here
        jax
        blackjax
        flax
        matplotlib
        pandas
        virtualenv
      ]);


    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          pythonEnv
          duckdb # Add other dev tools
          uv
        ];
        shellHook = ''

  if [ ! -d ".venv" ]; then
    uv venv .venv --python=${pythonEnv}/bin/python
  fi
  
  source .venv/bin/activate
  
  if [ -f "requirements.txt" ]; then
    uv pip sync requirements.txt
  fi

                    '';
      };
    };
}
