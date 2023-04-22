package main

import "os"

func main() {
	if len(os.Args) < 2 {
		os.Args = append(os.Args, "--help")
	}

}
