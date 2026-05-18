{
  description = "universal Triton kernel";
  inputs.kernel-builder.url = "github:huggingface/kernel-builder/b4accba4496b28faef19a0487fbcf9686b14e2ef";
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genFlakeOutputs { path = ./.; rev = self.shortRev or self.dirtyShortRev or "dev0"; doGetKernelCheck = false; };
}
