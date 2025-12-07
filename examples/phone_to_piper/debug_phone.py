import hebi
import time
import inspect

def main():
    print("Looking for HEBI Mobile I/O device...")
    lookup = hebi.Lookup()
    time.sleep(2.0)
    
    group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
    if group is None:
        print("[ERROR] Could not find device")
        return

    print("[SUCCESS] Connected to group!")
    
    try:
        while True:
            fbk = group.get_next_feedback(timeout_ms=100)
            if fbk is None:
                continue

            if fbk.io and fbk.io.b:
                try:
                    val = fbk.io.b.get_int(1)
                    print(f"Raw: {val} | Type: {type(val)} | Bool: {bool(val)}")
                    
                    # Simulate teleop_phone logic
                    # button_b1_pressed = bool(button_b.get_int(1))
                    # We want to see if this raises exception or returns True/False
                    
                    # Also check length if it is iterable
                    if hasattr(val, '__len__'):
                        print(f"Len: {len(val)} | Item 0: {val[0]}")
                        
                except Exception as e:
                    print(f"Error calling get_int(1): {e}")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    main()
