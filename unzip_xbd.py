import tarfile
import os
import glob

def extract_tar_gz(directory="."):
    """Finds all .tar.gz files in the directory and extracts them into xbd_data/"""
    target_dir = "xbd_data"
    os.makedirs(target_dir, exist_ok=True)
    
    tar_files = glob.glob(os.path.join(directory, "*.tar.gz"))
    
    if not tar_files:
        print("No .tar.gz files found in the current directory.")
        print("Make sure you downloaded the xView2 train/test files here!")
        return
        
    for file_path in tar_files:
        print(f"[*] Extracting {os.path.basename(file_path)} into {target_dir}/ ...")
        try:
            # Open and extract the tarball
            with tarfile.open(file_path, "r:gz") as tar:
                # Basic security check for malicious tarballs
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                    tar.extractall(path, members, numeric_owner=numeric_owner)

                safe_extract(tar, path=target_dir)
                
            print(f"[+] Complete: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"[-] Failed to extract {os.path.basename(file_path)}: {e}")

if __name__ == "__main__":
    print("Starting extraction script...")
    extract_tar_gz()
    print("All tasks finished.")
