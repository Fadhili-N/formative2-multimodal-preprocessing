"""Multimodal authentication and product recommendation CLI.

Flow:
    1. Face image  -> facial recognition model      unknown face   -> ACCESS DENIED
    2. Recognised  -> product model is called; the prediction is prepared but held back
    3. Voice clip  -> voiceprint verification model  unknown voice  -> ACCESS DENIED
                      voice of another member than the face          -> ACCESS DENIED
    4. Approved    -> the recommended product is displayed

Usage:
    python main.py                              interactive, asks for file paths
    python main.py --face IMG --voice AUDIO     single transaction, no prompts
    python main.py --demo                       runs every demo scenario in a row

Models used (each one lives in its owner's folder):
    face/face_model.py                      predict(image_path) -> dict
    audio/voice_model.py                    verify(audio_path)  -> (accepted, member, confidence)
    armstrong_product_model/product_model.py predict(customer_id) -> product category
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL_FOLDERS = {
    "face_model": "face",
    "voice_model": "audio",
    "product_model": "armstrong_product_model",
}
# Each folder is put on the import path so the scripts can be imported unchanged
# (audio/voice_model.py itself runs `from audio_pipeline import ...`).
for folder in MODEL_FOLDERS.values():
    sys.path.insert(0, str(ROOT / folder))

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXT = {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".mp4", ".flac"}

# The face and voice models were trained with slightly different labels for the
# same person; both are mapped to one canonical member name.
MEMBER_ALIASES = {"ayoo": "ayomide"}

# Simulated link between each team member and a customer profile.
# All four IDs exist in both customer_transactions.csv and customer_social_profiles.csv
# and are among the customers with the longest purchase history.
MEMBER_CUSTOMERS = {
    "armstrong": 187,
    "ayomide": 114,
    "fadhili": 101,
    "diane": 188,
}

DEMO_SCENARIOS = [
    ("Full transaction: Armstrong",
     "face/armstrong/armstrong_neutral.jpeg", "audio/converted/armstrong_yes_approve.wav"),
    ("Full transaction: Ayomide",
     "face/ayoo/ayoo_happy.jpg", "audio/converted/ayomide_confirm_transaction.wav"),
    ("Full transaction: Diane",
     "face/diane/diane_surprised.jpeg", "audio/converted/diane_yes_approve.wav"),
    ("Full transaction: Fadhili",
     "face/fadhili/fadhili_neutral.jpeg", "audio/converted/fadhili_confirm_transaction.wav"),
    ("Unauthorized face (stranger)",
     "face/test_images/smiling_test1.jpg", "audio/converted/diane_yes_approve.wav"),
    ("Unauthorized face (no face in image)",
     "face/test_images/unauthorized_no_face.jpg", "audio/converted/diane_yes_approve.wav"),
    ("Unauthorized voice (stranger)",
     "face/armstrong/armstrong_neutral.jpeg", "audio/unauthorized/stranger_tts2.wav"),
    ("Face and voice from different members",
     "face/ayoo/ayoo_neutral.jpeg", "audio/converted/fadhili_yes_approve.wav"),
]


# ------------------------------------------------------------------ terminal output
class C:
    on = sys.stdout.isatty()
    GREEN = "\033[92m" if on else ""
    RED = "\033[91m" if on else ""
    CYAN = "\033[96m" if on else ""
    BOLD = "\033[1m" if on else ""
    END = "\033[0m" if on else ""


def banner(text):
    line = "=" * 64
    print(f"\n{C.CYAN}{line}\n  {text}\n{line}{C.END}")


def step(n, text):
    print(f"\n{C.BOLD}[Step {n}] {text}{C.END}")


def ok(text):
    print(f"  {C.GREEN}✔ {text}{C.END}")


def fail(text):
    print(f"  {C.RED}✘ {text}{C.END}")


def info(text):
    print(f"  {text}")


def denied(reason):
    print(f"\n{C.RED}{C.BOLD}  ACCESS DENIED: {reason}{C.END}\n")
    return False


def canonical(member):
    return MEMBER_ALIASES.get(member, member) if member else member


def display(name):
    return name.capitalize() if name else "unknown"


# ------------------------------------------------------------------ model loading
class ModelUnavailable(Exception):
    pass


_modules = {}


def load(module_name, required_function):
    """Import a model script once, with a clear message if it cannot be used yet."""
    if module_name in _modules:
        return _modules[module_name]
    path = ROOT / MODEL_FOLDERS[module_name] / f"{module_name}.py"
    if not path.exists():
        raise ModelUnavailable(f"{path.relative_to(ROOT)} not found")
    # Checked before importing: a script without this function would run its
    # whole training code on import instead of exposing a callable model.
    if f"def {required_function}(" not in path.read_text(encoding="utf-8"):
        raise ModelUnavailable(f"{path.relative_to(ROOT)} has no {required_function}() function")
    try:
        _modules[module_name] = importlib.import_module(module_name)
    except ImportError as e:
        raise ModelUnavailable(f"cannot import {module_name}: {e} "
                               f"(run: pip install -r requirements.txt)") from e
    return _modules[module_name]


def resolve(path, allowed, kind):
    p = Path(path).expanduser()
    if not p.is_absolute() and (ROOT / p).exists():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"{kind} file not found: {path}")
    if p.suffix.lower() not in allowed:
        raise ValueError(f"unsupported {kind} format '{p.suffix}' "
                         f"(expected one of {', '.join(sorted(allowed))})")
    return p.resolve()


def shown(p):
    return p.relative_to(ROOT) if p.is_relative_to(ROOT) else p


# ------------------------------------------------------------------ one transaction
def run_transaction(face_path, voice_path=None, ask_voice=False):
    """One full authentication + recommendation attempt. Returns True if approved."""
    # Step 1: facial recognition
    step(1, "Facial recognition")
    try:
        face_file = resolve(face_path, IMAGE_EXT, "image")
        info(f"Input image: {shown(face_file)}")
        result = load("face_model", "predict").predict(str(face_file))
    except (FileNotFoundError, ValueError, ModelUnavailable) as e:
        return denied(f"face check could not run ({e})")

    if not result["authorized"]:
        fail(f"Face not recognised: {result['reason']}")
        return denied("unknown face")
    member = canonical(result["member"])
    ok(f"Face recognised: {display(member)} "
       f"(probability {result['probability']:.2f}, similarity {result['similarity']:.2f})")

    # Step 2: product prediction, prepared but only shown after voice approval
    step(2, "Product recommendation model")
    customer_id = MEMBER_CUSTOMERS.get(member)
    if customer_id is None:
        return denied(f"no customer profile linked to {display(member)}")
    try:
        product = load("product_model", "predict").predict(customer_id)
    except (ModelUnavailable, KeyError, ValueError, FileNotFoundError) as e:
        return denied(f"product model could not run ({e})")
    ok(f"Prediction prepared for customer {customer_id}. Waiting for voice confirmation.")

    # Step 3: voiceprint verification
    step(3, "Voiceprint verification")
    if ask_voice:
        voice_path = prompt("  Path to voice sample (e.g. 'Yes, approve'): ")
    try:
        voice_file = resolve(voice_path, AUDIO_EXT, "audio")
        info(f"Input audio: {shown(voice_file)}")
        voice_ok, voice_member, voice_conf = load("voice_model", "verify").verify(str(voice_file))
    except (FileNotFoundError, ValueError, ModelUnavailable) as e:
        return denied(f"voice check could not run ({e})")

    voice_member = canonical(voice_member)
    if not voice_ok:
        fail(f"Voice not approved (closest: {display(voice_member)}, confidence {voice_conf:.2f})")
        return denied("unknown voice")
    if voice_member != member:
        fail(f"Voice belongs to {display(voice_member)}, face belongs to {display(member)}")
        return denied("voice does not match the recognised face")
    ok(f"Voice approved: {display(voice_member)} (confidence {voice_conf:.2f})")

    # Step 4: result
    step(4, "Result")
    print(f"\n{C.GREEN}{C.BOLD}  TRANSACTION APPROVED for {display(member)}")
    print(f"  Recommended product: {product}{C.END}\n")
    return True


# ------------------------------------------------------------------ modes
def prompt(text):
    try:
        return input(text).strip().strip("'\"")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        sys.exit(0)


def interactive():
    banner("Multimodal Authentication & Product Recommendation")
    print("  Type 'q' at any prompt to quit.")
    while True:
        face = prompt("\n  Path to face image: ")
        if face.lower() in {"q", "quit", "exit"}:
            break
        run_transaction(face, ask_voice=True)
        if prompt("  Run another transaction? [y/N]: ").lower() != "y":
            break
    print("Goodbye.")


def demo():
    results = []
    for i, (title, face, voice) in enumerate(DEMO_SCENARIOS, 1):
        banner(f"Scenario {i}/{len(DEMO_SCENARIOS)}: {title}")
        results.append((title, run_transaction(face, voice)))
        time.sleep(0.5)
    banner("Demo summary")
    for title, approved in results:
        status = f"{C.GREEN}APPROVED{C.END}" if approved else f"{C.RED}DENIED{C.END}"
        print(f"  {title:<45} {status}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Face -> voice -> product recommendation CLI")
    parser.add_argument("--face", help="path to a face image")
    parser.add_argument("--voice", help="path to a voice sample")
    parser.add_argument("--demo", action="store_true", help="run all demo scenarios")
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.face:
        banner("Multimodal Authentication & Product Recommendation")
        sys.exit(0 if run_transaction(args.face, args.voice, ask_voice=args.voice is None) else 1)
    else:
        interactive()


if __name__ == "__main__":
    main()
