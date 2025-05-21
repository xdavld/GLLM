import time

print("Starting dummy job...")
for i in range(5):
    print(f"Step {i+1}/5: sleeping for 2 seconds...")
    time.sleep(2)
print("Dummy job finished.")