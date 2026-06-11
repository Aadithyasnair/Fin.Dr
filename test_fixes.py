"""FIN.DR - Fix Verification Tests"""
import requests, json, time

BASE = "http://localhost:5000"

def post(payload):
    return requests.post(f"{BASE}/api/predict", json=payload).json()

print("\n" + "="*60)
print("TEST 1: Large Electronics Purchase at Croma (purchase type)")
print("="*60)
d = post({"amount": 89999, "merchant": "Croma Electronics", "user_id": "USR001",
          "txn_type": "purchase", "card_type": "VISA", "location": "Bangalore"})
s = d["scores"]; f = d["fusion"]
print(f"  TF:{s['tf_score']:.2%}  Transformer:{s['transformer_score']:.2%}  Anomaly:{s['anomaly_score']:.2%}")
print(f"  Final: {f['final_score']:.2%}  =>  {f['decision']}")
print(f"  Context: {f.get('context_note', '-')}")
print(f"  Reason: {f['reason']}")

print("\n" + "="*60)
print("TEST 2: Large Self-Transfer to Own HDFC Account")
print("="*60)
d = post({"amount": 150000, "merchant": "HDFC Bank NEFT to Self", "user_id": "USR001",
          "txn_type": "self_transfer", "card_type": "RuPay", "location": "Mumbai"})
s = d["scores"]; f = d["fusion"]
print(f"  TF:{s['tf_score']:.2%}  Transformer:{s['transformer_score']:.2%}  Anomaly:{s['anomaly_score']:.2%}")
print(f"  Final: {f['final_score']:.2%}  =>  {f['decision']}")
print(f"  Context: {f.get('context_note', '-')}")
print(f"  Reason: {f['reason']}")

print("\n" + "="*60)
print("TEST 3: High-risk Unknown Offshore Merchant (should still BLOCK)")
print("="*60)
d = post({"amount": 4899, "merchant": "Unknown Offshore Store", "user_id": "USR002",
          "txn_type": "purchase", "card_type": "MASTERCARD", "location": "Unknown"})
s = d["scores"]; f = d["fusion"]
print(f"  TF:{s['tf_score']:.2%}  Transformer:{s['transformer_score']:.2%}  Anomaly:{s['anomaly_score']:.2%}")
print(f"  Final: {f['final_score']:.2%}  =>  {f['decision']}")

print("\n" + "="*60)
print("TEST 4: Amazon MacBook Purchase (trusted merchant keyword)")
print("="*60)
d = post({"amount": 149999, "merchant": "Amazon India - MacBook Pro", "user_id": "USR001",
          "txn_type": "purchase", "card_type": "VISA", "location": "Delhi"})
s = d["scores"]; f = d["fusion"]
print(f"  TF:{s['tf_score']:.2%}  Transformer:{s['transformer_score']:.2%}  Anomaly:{s['anomaly_score']:.2%}")
print(f"  Final: {f['final_score']:.2%}  =>  {f['decision']}")
print(f"  Context: {f.get('context_note', '-')}")

print("\n" + "="*60)
print("TEST 5: EMI Payment (bill_payment type)")
print("="*60)
d = post({"amount": 25000, "merchant": "HDFC Bank EMI", "user_id": "USR001",
          "txn_type": "emi", "card_type": "VISA", "location": "Mumbai"})
s = d["scores"]; f = d["fusion"]
print(f"  TF:{s['tf_score']:.2%}  Transformer:{s['transformer_score']:.2%}  Anomaly:{s['anomaly_score']:.2%}")
print(f"  Final: {f['final_score']:.2%}  =>  {f['decision']}")
print(f"  Context: {f.get('context_note', '-')}")

print("\n" + "="*60)
print("TEST 6: Voice Verification Keyboard Input Fix")
print("="*60)
# Force a CALL_USER scenario via medium-risk amount
d = post({"amount": 2800, "merchant": "Suspicious Shop", "user_id": "USR099",
          "txn_type": "purchase", "card_type": "VISA"})
f = d["fusion"]; txn_id = d["transaction_id"]
print(f"  Decision: {f['decision']}  Score: {f['final_score']:.2%}  TXN: {txn_id}")

if f["decision"] == "CALL_USER":
    rv = requests.post(f"{BASE}/api/voice/start", json={
        "transaction_id": txn_id, "amount": 2800,
        "merchant": "Suspicious Shop", "timestamp": "now"
    })
    print(f"  Voice trigger: {rv.json().get('phase')} question='{rv.json().get('question')}'")
    time.sleep(1)

    # Inject keyboard response (keyboard fallback forwards to responder)
    ri = requests.post(f"{BASE}/api/voice/input/{txn_id}", json={"text": "yes"})
    print(f"  Keyboard inject: success={ri.json().get('success')}")

    # Wait and poll
    for i in range(8):
        time.sleep(2)
        rs = requests.get(f"{BASE}/api/voice/status/{txn_id}").json()
        print(f"  Poll {i+1}: phase={rs.get('phase')}  status={rs.get('status','')}")
        if rs.get("phase") == "COMPLETE":
            print(f"  FINAL Voice Result: {rs.get('status')} — {rs.get('reason')}")
            break
else:
    print(f"  Transaction was {f['decision']} — adjust amount for CALL_USER range")

print("\n" + "="*60)
print("All tests complete!")
print("="*60)
