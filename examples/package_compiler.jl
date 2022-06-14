using PackageCompiler
execution_path = pwd() * "/examples/precompile_fouriercore.jl"
# execution_path = "precompile_fouriercore.jl"
create_sysimage(["FourierCore"], sysimage_path="fouriercore.so", precompile_execution_file=execution_path)