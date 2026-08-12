target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")

    -- Runtime-only .cu files contain host-side CUDA Runtime API calls.  Build
    -- them as ordinary C++ so this target also works with CoreX images whose
    -- `nvcc` executable is only a compatibility/version shim.  Actual kernels
    -- use the vendor compiler in the GPU-ops target.
    add_files("../src/device/nvidia/*.cu", {
        sourcekind = "cxx",
    })
    add_cxflags("-fPIC", {force = true})
    add_includedirs("/usr/local/corex/include")
    add_linkdirs("/usr/local/corex/lib64", {public = true})
    add_links("cudart", {public = true})

    on_install(function (target) end)
target_end()

rule("corex.ivcore")
    set_extensions(".corex")
    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        local objectfile = target:objectfile(sourcefile)
        local found = false
        for _, existing in ipairs(target:objectfiles()) do
            if existing == objectfile then
                found = true
                break
            end
        end
        if not found then
            table.insert(target:objectfiles(), objectfile)
        end
        batchcmds:show_progress(opt.progress, "${color.build.object}ivcore %s", sourcefile)
        batchcmds:mkdir(path.directory(objectfile))
        batchcmds:vrunv("/usr/local/corex/bin/clang++", {
            "-c", "-O3", "-std=c++20", "-fPIC", "-x", "ivcore",
            "--cuda-gpu-arch=ivcore11", "--cuda-path=/usr/local/corex",
            "-Iinclude", "-Isrc", "-o", objectfile, sourcefile,
        })
        batchcmds:add_depfiles(sourcefile)
        batchcmds:set_depmtime(os.mtime(objectfile))
        batchcmds:set_depcache(target:dependfile(objectfile))
    end)
rule_end()

target("llaisys-ops-nvidia")
    set_kind("object")
    add_deps("llaisys-tensor")
    set_languages("cxx20")
    add_files("../src/ops/nvidia/*.corex", {rules = "corex.ivcore"})
    add_linkdirs("/usr/local/corex/lib64", {public = true})
    add_links("cudart", "cublas", {public = true})

    on_install(function (target) end)
target_end()
