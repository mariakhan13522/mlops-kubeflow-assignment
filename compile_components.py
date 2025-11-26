"""
Component Compiler Script
Compiles Kubeflow components to YAML files

"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from kfp import compiler
from pipeline_components import (
    data_extraction,
    data_preprocessing,
    model_training,
    model_evaluation
)

def compile_components():
    """
    Compile all components to YAML files in components/ directory
    """
    
    # Create components directory if it doesn't exist
    os.makedirs('components', exist_ok=True)
    
    print("=" * 70)
    print("COMPILING KUBEFLOW COMPONENTS TO YAML")
    print("=" * 70)
    
    # List of components to compile
    components_list = [
        ('data_extraction', data_extraction),
        ('data_preprocessing', data_preprocessing),
        ('model_training', model_training),
        ('model_evaluation', model_evaluation)
    ]
    
    compiled_count = 0
    
    # Compile each component
    for component_name, component_func in components_list:
        output_file = f'components/{component_name}.yaml'
        
        print(f"\n✓ Compiling: {component_name}")
        print(f"  Output: {output_file}")
        
        try:
            # Compile component to YAML
            compiler.Compiler().compile(
                pipeline_func=component_func,
                package_path=output_file
            )
            
            # Check file size
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"  Status: ✓ SUCCESS")
                print(f"  Size: {file_size:,} bytes")
                compiled_count += 1
            else:
                print(f"  Status: ✗ FAILED - File not created")
                
        except Exception as e:
            print(f"  Status: ✗ ERROR")
            print(f"  Error: {str(e)}")
    
    print("\n" + "=" * 70)
    print(f"COMPILATION COMPLETE: {compiled_count}/4 components compiled")
    print("=" * 70)
    
    if compiled_count == 4:
        print("\n✓ All components compiled successfully!")
        print("\nGenerated YAML files in components/ directory:")
        for component_name, _ in components_list:
            print(f"  - components/{component_name}.yaml")
        return True
    else:
        print(f"\n✗ Only {compiled_count} components compiled successfully")
        print("  Please check errors above")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("KUBEFLOW COMPONENT COMPILER")
    print("=" * 70)
    print("Author: Maria Khan")
    print("Assignment: Cloud MLOps #4")
    print("=" * 70 + "\n")
    
    success = compile_components()
    
    if success:
        print("\n" + "=" * 70)
        print("✓ COMPILATION SUCCESSFUL!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Check components/ directory for YAML files")
        print("2. These YAML files can be reused in any Kubeflow pipeline")
        print("3. Commit all files to Git")
    else:
        print("\n" + "=" * 70)
        print("✗ COMPILATION FAILED")
        print("=" * 70)
        sys.exit(1)