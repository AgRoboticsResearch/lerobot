import hebi
import inspect

def main():
    lookup = hebi.Lookup()
    print("Lookup created")
    # Just creating a mock group if possible or inspecting lookup
    # Actually finding a group might take time/fail if no device.
    # But hebi usually exposes classes.
    
    print(f"hebi content: {dir(hebi)}")
    
    # Try to find where Group is defined
    if hasattr(hebi, 'Group'):
        print("Found hebi.Group")
        print(inspect.signature(hebi.Group.get_next_feedback))
    else:
        print("hebi.Group not found directly")

    # If we can't find Group, let's try to see if we can get it from lookup mock
    # (Not easy without device). 
    
    # Let's try to print help on the module
    # help(hebi) 

if __name__ == "__main__":
    main()
