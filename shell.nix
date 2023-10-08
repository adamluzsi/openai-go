{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.docker
    pkgs.docker-compose
    pkgs.go
  ];


  shellHook = ''
    export GOPATH=''${GOPATH:-''$HOME/go}
    export PATH=''${PATH}:''${GOPATH}/bin
  '';
}

