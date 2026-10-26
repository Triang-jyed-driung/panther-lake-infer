// Compile an OpenVINO IR model and stream the compiled device blob to disk.
//
// Usage:
//   compile_blob_cpp model.xml [NPU|GPU] [output.blob]
//
// Build against the OpenVINO installation used by Python:
//   OV_PY=$(python -c 'import openvino, pathlib; print(pathlib.Path(openvino.__file__).parent)')
//   g++ -std=c++17 -O2 -I"$OV_PY/include" scripts/npu/compile_blob.cpp \
//       "$OV_PY"/libs/libopenvino.so.* -Wl,-rpath,"$OV_PY/libs" \
//       -o scripts/npu/compile_blob_cpp
//
// Using CompiledModel::export_model(std::ostream&) avoids buffering a large
// compiled model in a Python BytesIO object before writing it to disk.

#include <openvino/openvino.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0]
                  << " model.xml [NPU|GPU] [output.blob]\n";
        return 2;
    }

    const fs::path xml = fs::absolute(argv[1]);
    const std::string device = argc >= 3 ? argv[2] : "NPU";
    if (device != "NPU" && device != "GPU") {
        std::cerr << "Device must be NPU or GPU\n";
        return 2;
    }
    const fs::path output = argc >= 4
        ? fs::path(argv[3])
        : xml.parent_path() / (xml.stem().string() + ".blob");
    const fs::path temporary = output.string() + ".tmp";

    try {
        std::cout << "Reading " << xml << "\n";
        ov::Core core;
        auto model = core.read_model(xml.string());

        std::cout << "Compiling for " << device << "...\n" << std::flush;
        auto compiled = core.compile_model(model, device);

        if (!output.parent_path().empty()) {
            fs::create_directories(output.parent_path());
        }
        {
            std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
            if (!stream) {
                throw std::runtime_error("Cannot open output file: " + temporary.string());
            }
            std::cout << "Exporting directly to " << temporary << "...\n" << std::flush;
            compiled.export_model(stream);
            stream.close();
            if (!stream) {
                throw std::runtime_error("Failed writing output file: " + temporary.string());
            }
        }

        fs::rename(temporary, output);
        std::cout << output << " (" << fs::file_size(output) << " bytes)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Compile/export failed: " << e.what() << "\n";
        return 1;
    }
}
