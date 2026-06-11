import speech_recognition as sr

print("\n=== MICROPHONES ===")
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(i, name)
print("===================\n")