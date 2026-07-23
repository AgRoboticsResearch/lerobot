#!/usr/bin/env python
r"""Standalone DM4310 gripper control — type an angle in degrees, it moves there.

This is a minimal, arm-free tool that reuses the exact same Gripper class the
deploy script (deploy_umi_relative_ee_piper.py) uses. It connects to the
external DM4310 gripper over the DAMIAO DM-FDCAN serial bridge, then reads
target angles (degrees) from stdin and sends them as MIT-mode impedance
commands (radians).

How the deploy script controls the gripper (for reference):
    from modules.gripper import Gripper
    gripper = Gripper(port="/dev/ttyACM0")
    gripper.connect()
    gripper.send_command(kp=5.0, kd=0.5, position=<radians>)  # move
    pos_rad = gripper.position                                  # read

Degrees -> radians conversion:  rad = deg * pi / 180
Working range (from deploy script):  OPEN = -0.139 rad (~-7.96°)
                                     CLOSED =  0.734 rad (~ 42.05°)

Usage:
    python gripper_control.py
    python gripper_control.py --port /dev/ttyACM0 --kp 5.0 --kd 0.5
    python gripper_control.py --no-clamp        # allow angles beyond working range
    python gripper_control.py --move 20         # non-interactive: move to 20°, then exit

Interactive commands (just type and press Enter):
    <number>   move to <number> degrees (prints current + asks to confirm first)
    open / o   move to the open limit
    close / c  move to the closed limit
    home / hm  close against the mechanical stop, then zero (discovers range)
    read / r   print current motor state
    zero / z   set the current position as 0 rad (encoder zero)
    quit / q   disable motor and exit  (Ctrl+C works too)

In a live terminal the current angle is shown on a status line refreshed at
--hz (default 1 Hz; use e.g. --hz 20 for a fast readout). Each refresh re-sends
the current hold target, because the DM4310 only reports state in reply to a
command — that reply is what keeps the readout live while you decide what to
type and while the gripper moves. Every move prints current + target + delta
and waits for Enter unless --yes is given — so you never send a command blind.
If you swapped the gripper, run 'home' first to learn the new range.
"""

import argparse
import math
import os
import select
import sys
import termios
import tty


class _QuitLoop(Exception):
    """Internal sentinel raised to exit the live loop cleanly."""

# Make modules.gripper importable (same path trick as the deploy script).
DEFAULT_PIPER_SRC_PATH = os.path.normpath(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "sroi-piper", "src",
)))

# Working range in radians, copied from deploy_umi_relative_ee_piper.py.
GRIPPER_OPEN_RAD = 0
GRIPPER_CLOSED_RAD = -0.91


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone DM4310 gripper control (input degrees to move)"
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0",
                        help="Serial port of the DM-FDCAN bridge")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--can_id", type=lambda s: int(s, 0), default=0x08,
                        help="Motor CAN ID (Change in DM Debug Tool hex ok, e.g. 0x08)")
    parser.add_argument("--recv_id", type=lambda s: int(s, 0), default=0x18,
                        help="Motor receive CAN ID (Master ID in DM Debug Tool hex ok, e.g. 0x18)")
    parser.add_argument("--kp", type=float, default=5.0,
                        help="Position stiffness gain (MIT mode, 0..500)")
    parser.add_argument("--kd", type=float, default=0.5,
                        help="Damping gain (MIT mode, 0..5)")
    parser.add_argument("--open_rad", type=float, default=GRIPPER_OPEN_RAD,
                        help="Open-limit angle in radians")
    parser.add_argument("--closed_rad", type=float, default=GRIPPER_CLOSED_RAD,
                        help="Closed-limit angle in radians")
    parser.add_argument("--no_clamp", action="store_true",
                        help="Do NOT clamp targets to the open/closed range")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the pre-move confirmation prompt (send immediately)")
    parser.add_argument("--hz", type=float, default=1.0,
                        help="Live readout update rate in Hz (default 1.0). Typing "
                             "stays responsive regardless — this only sets how often "
                             "the status line refreshes when idle.")
    parser.add_argument("--piper_src", type=str, default=DEFAULT_PIPER_SRC_PATH,
                        help="Path to sroi-piper/src (for modules.gripper)")
    parser.add_argument("--move", type=float, default=None,
                        help="Non-interactive: move to this angle (degrees) and exit")
    return parser.parse_args()


def connect_gripper(args: argparse.Namespace):
    if args.piper_src not in sys.path:
        sys.path.insert(0, args.piper_src)
    from modules.gripper import Gripper  # noqa: E402

    gripper = Gripper(
        port=args.port,
        baudrate=args.baudrate,
        can_id=args.can_id,
        recv_id=args.recv_id,
    )
    gripper.connect()
    return gripper


def clamp_rad(rad: float, args: argparse.Namespace) -> float:
    if args.no_clamp:
        return rad
    lo, hi = min(args.open_rad, args.closed_rad), max(args.open_rad, args.closed_rad)
    return max(lo, min(hi, rad))


def probe_state(gripper, args):
    """Solicit a FRESH state frame from the motor.

    The DM4310 over the DM-FDCAN bridge only reports state in reply to a
    command — it does not stream on its own — so a bare read_state() returns
    whatever the last command's reply was (possibly stale). Re-sending a hold
    at the current position forces an up-to-date reply.
    """
    return gripper.send_command(kp=args.kp, kd=args.kd, position=gripper.position)


def send_target(gripper, target_rad: float, args) -> str:
    """Send an already-clamped target (radians); return a one-line log message."""
    current_rad = probe_state(gripper, args).position
    delta_deg = rad_to_deg(target_rad) - rad_to_deg(current_rad)
    state = gripper.send_command(kp=args.kp, kd=args.kd, position=target_rad)
    return (f"current {rad_to_deg(current_rad):+7.2f}° -> "
            f"target {rad_to_deg(target_rad):+7.2f}° (delta {delta_deg:+.2f}°) "
            f"-> actual {rad_to_deg(state.position):+7.2f}°")


def apply_move(gripper, deg: float, args) -> str:
    """Clamp `deg` to the working range, send it, and return a log message.

    No prompting — the caller decides whether to confirm first.
    """
    requested_rad = deg_to_rad(deg)
    target_rad = clamp_rad(requested_rad, args)
    prefix = ""
    if not args.no_clamp and abs(target_rad - requested_rad) > 1e-6:
        lo_deg = rad_to_deg(min(args.open_rad, args.closed_rad))
        hi_deg = rad_to_deg(max(args.open_rad, args.closed_rad))
        prefix = (f"[clamped {deg:.2f}° -> {rad_to_deg(target_rad):.2f}°; "
                  f"range {lo_deg:.2f}°..{hi_deg:.2f}° from prev gripper] ")
    return prefix + send_target(gripper, target_rad, args)


def send_deg(gripper, deg: float, args, *, confirm: bool = True) -> None:
    """Blocking-input move: show current angle, (optionally) confirm, then send.

    Used by the fallback `interactive_loop` and `--move`. The working range may
    differ on a freshly installed gripper, so this never sends blindly: it reads
    + prints the current position and the delta first, and (unless --yes) waits
    for confirmation before actually sending.
    """
    target_rad = clamp_rad(deg_to_rad(deg), args)
    if confirm and not args.yes:
        current_rad = probe_state(gripper, args).position
        delta_deg = rad_to_deg(target_rad) - rad_to_deg(current_rad)
        print(f"  current {rad_to_deg(current_rad):+7.2f}°  ->  "
              f"target {rad_to_deg(target_rad):+7.2f}°  (delta {delta_deg:+.2f}°)")
        try:
            ans = input("  send? [Enter=yes / n=abort] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  aborted, no command sent")
            return
        if ans.startswith("n"):
            print("  aborted, no command sent")
            return
    print("  " + apply_move(gripper, deg, args))


def print_state(gripper, args) -> None:
    state = probe_state(gripper, args)
    print(f"  pos {rad_to_deg(state.position):+7.2f}° ({state.position:+.4f} rad) | "
          f"vel {state.velocity:+.3f} rad/s | torque {state.torque:+.3f} Nm | "
          f"MOS {state.temp_mos}°C rotor {state.temp_rotor}°C | status {state.status:#x}")


HELP_TEXT = """
Commands:
  <number>   move to <number> degrees  (shows current + asks to confirm first)
  open / o   move to open limit
  close / c  move to closed limit
  home / hm  close against the mechanical stop, then zero there (discovers range)
  read / r   print current motor state
  zero / z   set current position as encoder zero
  help / h   show this help
  quit / q   disable motor and exit  (Ctrl+C also works)
""".strip()


def interactive_loop(gripper, args) -> None:
    open_deg = rad_to_deg(args.open_rad)
    closed_deg = rad_to_deg(args.closed_rad)
    print(f"\nGripper connected on {args.port} (kp={args.kp}, kd={args.kd})")
    if not args.no_clamp:
        print(f"NOTE: clamp range open {open_deg:.2f}° .. closed {closed_deg:.2f}° "
              f"is from the PREVIOUS gripper. If this is a new gripper, run "
              f"'home' to discover its real range before trusting these limits.")
    print(f"Current position: ", end="")
    print_state(gripper, args)
    print(HELP_TEXT)

    while True:
        try:
            line = input("\ndeg> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        cmd = line.lower()
        try:
            if cmd in ("quit", "q", "exit"):
                break
            elif cmd in ("help", "h", "?"):
                print(HELP_TEXT)
            elif cmd in ("open", "o"):
                send_deg(gripper, open_deg, args)
            elif cmd in ("close", "c"):
                send_deg(gripper, closed_deg, args)
            elif cmd in ("read", "r", "status"):
                print_state(gripper, args)
            elif cmd in ("home", "hm"):
                print("  Homing: closing against the mechanical stop, then zeroing...")
                stop_state = gripper.home(kp=args.kp, kd=args.kd)
                print(f"  Homed. Travelled {rad_to_deg(stop_state.position):+.2f}° "
                      f"to the stop (now zeroed there). closed = 0°, open = -range.")
            elif cmd in ("zero", "z"):
                gripper.set_zero()
                print("  Encoder zeroed at current position.")
            else:
                deg = float(line)
                send_deg(gripper, deg, args)
        except ValueError:
            print(f"  Not a number or command: {line!r}  (type 'h' for help)")
        except Exception as e:
            print(f"  Error: {e}")


def live_loop(gripper, args) -> None:
    """Real-time mode: a status line refreshed at --hz (default 1 Hz) while you
    type. Each refresh re-sends the current hold target, because the DM4310
    only reports state in reply to a command — that reply is what keeps the
    displayed angle live (you watch it settle after a move, and the value is
    fresh the instant you decide what to type). Typing stays responsive at any
    rate: select() wakes on the keypress itself.

    Uses cbreak stdin + select() so input and rendering share one non-blocking
    loop instead of a blocking input() that would freeze the display.
    """
    open_deg = rad_to_deg(args.open_rad)
    closed_deg = rad_to_deg(args.closed_rad)

    print(f"\nGripper connected on {args.port} (kp={args.kp}, kd={args.kd})")
    if not args.no_clamp:
        print(f"NOTE: clamp range open {open_deg:.2f}° .. closed {closed_deg:.2f}° "
              f"is from the PREVIOUS gripper — run 'home' to learn the new range.")
    print(f"Live mode: the angle below updates continuously ({args.hz:g} Hz).")
    print("Type a number + Enter to move (it asks y/n first), or a command:")
    print("  open | close | home | read | zero | help | quit   (Ctrl+C to exit)")
    sys.stdout.flush()

    # Idle refresh period; typing is NOT limited by this — select() wakes the
    # instant a key arrives, this only sets how often we redraw when idle.
    period = 1.0 / max(args.hz, 0.1)

    state = {
        "buf": "",
        "mode": "input",
        "target_rad": 0.0,
        # Last commanded target — what we keep re-sending so the motor keeps
        # replying with fresh state. Init to the current position: hold where
        # it is, no jump on start.
        "hold_rad": gripper.position,
    }

    def log(msg: str) -> None:
        # Persistent line: wipe the live status, print msg on its own line.
        sys.stdout.write("\r\033[K" + msg + "\n")
        sys.stdout.flush()

    def read_key(timeout: float):
        # Read the fd directly (os.read, 1 byte) instead of sys.stdin.read(1):
        # the latter is buffered and can pull several bytes into Python's
        # internal buffer at once, after which select() on the fd no longer
        # sees them — stranding keys (e.g. the \n after a pasted "20").
        fd = sys.stdin.fileno()
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None
        try:
            data = os.read(fd, 1)
        except OSError:
            return ""
        return data.decode("utf-8", "replace")  # "" == EOF

    def drain_escape() -> None:
        """Swallow the rest of an arrow-key / escape sequence."""
        while read_key(0.01) is not None:
            pass

    def redraw() -> None:
        # Stream the hold target every tick: the DM4310 only reports state in
        # reply to a command, so THIS is what keeps the readout live. The reply
        # carries the fresh measured position/velocity/torque.
        s = gripper.send_command(kp=args.kp, kd=args.kd, position=state["hold_rad"])
        status = (f"pos {rad_to_deg(s.position):+7.2f}° ({s.position:+.4f} rad) | "
                  f"vel {s.velocity:+.2f} | τ {s.torque:+.2f}Nm | "
                  f"MOS {s.temp_mos}°C rotor {s.temp_rotor}°C | st {s.status:#x}")
        if state["mode"] == "input":
            prompt = f"  | deg> {state['buf']}"
        else:
            prompt = f"  | send {rad_to_deg(state['target_rad']):+7.2f}°? [y/n]"
        sys.stdout.write("\r\033[K" + status + prompt)
        sys.stdout.flush()

    def on_enter() -> None:
        text = state["buf"].strip()
        state["buf"] = ""
        if text == "":
            return
        low = text.lower()
        if low in ("quit", "q", "exit"):
            raise _QuitLoop()
        if low in ("help", "h", "?"):
            log("  " + HELP_TEXT.replace("\n", "\n  "))
            return
        if low in ("read", "r", "status"):
            s = probe_state(gripper, args)
            log(f"  pos {rad_to_deg(s.position):+7.2f}° ({s.position:+.4f} rad) | "
                f"vel {s.velocity:+.3f} | τ {s.torque:+.3f}Nm | MOS {s.temp_mos}°C "
                f"rotor {s.temp_rotor}°C | status {s.status:#x}")
            return
        if low in ("zero", "z"):
            gripper.set_zero()
            state["hold_rad"] = gripper.position  # fresh post-zero reply (~0)
            log("  Encoder zeroed at current position.")
            return
        if low in ("open", "o"):
            log("  " + apply_move(gripper, open_deg, args))
            state["hold_rad"] = args.open_rad
            return
        if low in ("close", "c"):
            log("  " + apply_move(gripper, closed_deg, args))
            state["hold_rad"] = args.closed_rad
            return
        if low in ("home", "hm"):
            log("  Homing: closing against the mechanical stop, then zeroing...")
            stop_state = gripper.home(kp=args.kp, kd=args.kd)
            state["hold_rad"] = gripper.position  # fresh post-zero reply (~0)
            log(f"  Homed. Travelled {rad_to_deg(stop_state.position):+.2f}° to the "
                f"stop (now zeroed there). closed = 0°, open = -range.")
            return
        try:
            deg = float(text)
        except ValueError:
            log(f"  Not a number or command: {text!r}  (type 'h' for help)")
            return
        if args.yes:
            log("  " + apply_move(gripper, deg, args))
            state["hold_rad"] = clamp_rad(deg_to_rad(deg), args)
        else:
            state["target_rad"] = clamp_rad(deg_to_rad(deg), args)
            state["mode"] = "confirm"

    def confirm_send() -> None:
        log("  " + send_target(gripper, state["target_rad"], args))
        state["hold_rad"] = state["target_rad"]
        state["mode"] = "input"

    def confirm_abort() -> None:
        state["mode"] = "input"
        log("  aborted, no command sent")

    # ── cbreak terminal setup (restored in finally) ────────────────────────
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd) if sys.stdin.isatty() else None
    if old is not None:
        tty.setcbreak(fd)
    try:
        while True:
            redraw()
            ch = read_key(period)
            if ch is None:                         # select timeout
                continue
            if ch == "" or ch == "\x04":           # EOF / Ctrl+D
                raise _QuitLoop()
            if ch == "\x03":                       # Ctrl+C
                raise KeyboardInterrupt
            if ch in ("\r", "\n"):                 # Enter
                if state["mode"] == "input":
                    on_enter()
                else:
                    confirm_send()
            elif ch in ("\x7f", "\b"):              # Backspace
                state["buf"] = state["buf"][:-1]
            elif ch == "\x1b":                      # Esc / arrow key
                if state["mode"] == "confirm":
                    confirm_abort()
                else:
                    drain_escape()
            elif state["mode"] == "confirm":
                if ch.lower() == "y":
                    confirm_send()
                elif ch.lower() == "n":
                    confirm_abort()
            elif ch.isprintable():
                state["buf"] += ch
    except (KeyboardInterrupt, _QuitLoop):
        pass
    finally:
        if old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()  # leave the live status line on its own line


def main() -> None:
    args = parse_args()

    try:
        gripper = connect_gripper(args)
    except Exception as e:
        print(f"Failed to connect to gripper on {args.port}: {e}", file=sys.stderr)
        print("Check: 24V power, CAN wiring, serial port, and --port/--can_id.", file=sys.stderr)
        sys.exit(1)

    try:
        if args.move is not None:
            # Non-interactive: still prints current + target first, but no prompt.
            send_deg(gripper, args.move, args, confirm=False)
        elif sys.stdin.isatty():
            live_loop(gripper, args)
        else:
            interactive_loop(gripper, args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            gripper.disconnect()
        except Exception:
            pass
        print("Gripper disabled and disconnected.")


if __name__ == "__main__":
    main()
