import os
import subprocess
import sys

# Set the base directory to search for shader files
base_dir = os.path.dirname(os.path.realpath(__file__))

# Set the path to the glslc compiler
glslc_path = "C:/VulkanSDK/1.3.290.0/Bin/glslc.exe"

# List of shader extensions and their corresponding output suffixes
shader_types = {
    ".frag": "-f.spv",
    ".vert": "-v.spv",
    ".geom": "-g.spv",
    ".tesc": "-tc.spv",
    ".tese": "-te.spv",
    ".comp": "-c.spv",
}


# Function to compile a shader file
def compile_shader(shader_path, output_path):
    command = [glslc_path, shader_path, "-o", output_path]
    try:
        subprocess.run(command, check=True)
        print(f"Successfully compiled {shader_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling {shader_path}: {e}")

# Recursively search for shader files and compile them
for root, _, files in os.walk(base_dir):
    for file in files:
        for extension, suffix in shader_types.items():
            if file.endswith(extension):
                shader_path = os.path.join(root, file)
                output_path = os.path.join(root, file.replace(extension, suffix))
                compile_shader(shader_path, output_path)
                break
