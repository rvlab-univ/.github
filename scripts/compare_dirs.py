import filecmp
import os

dir1 = '/home/lim/yolo_sam2/yolo_sam2_ws/tools/yolo11/X-AnyLabeling'
dir2 = '/home/lim/yolo_sam2/yolo_sam2_ws/tools/yolo11/X-AnyLabeling-New'
dir3 = '/home/lim/yolo_sam2/yolo_sam2_ws/tools/X-AnyLabeling'

def compare_dirs(d1, d2):
    if not os.path.exists(d1) or not os.path.exists(d2):
        print(f"One of the directories does not exist: {d1} or {d2}")
        return
    
    dcmp = filecmp.dircmp(d1, d2)
    diff_files = []
    
    def print_diff_files(dcmp):
        for name in dcmp.diff_files:
            diff_files.append(f"Differing file: {name} in {dcmp.left} and {dcmp.right}")
        for name in dcmp.left_only:
            diff_files.append(f"Only in {dcmp.left}: {name}")
        for name in dcmp.right_only:
            diff_files.append(f"Only in {dcmp.right}: {name}")
        for sub_dcmp in dcmp.subdirs.values():
            print_diff_files(sub_dcmp)
            
    print_diff_files(dcmp)
    if not diff_files:
        print(f"Directories {d1} and {d2} are identical.")
    else:
        print(f"Differences between {d1} and {d2}:")
        for f in diff_files[:50]: # limit output
            print(f)
        if len(diff_files) > 50:
            print(f"... and {len(diff_files) - 50} more differences.")

print("--- Comparing yolo11/X-AnyLabeling and yolo11/X-AnyLabeling-New ---")
compare_dirs(dir1, dir2)
print("\n--- Comparing yolo11/X-AnyLabeling and tools/X-AnyLabeling ---")
compare_dirs(dir1, dir3)
