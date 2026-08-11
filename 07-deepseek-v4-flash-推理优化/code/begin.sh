PROFILE_DIR="/public/home/aceoxvogz0/zyf/for_customer/hytest/prof"
PROFILE_START_STEP=10
PROFILE_NUM_STEPS=200

curl -X POST "http://10.5.5.78:30005/start_profile" \
  -H "Content-Type: application/json" \
  -d "{
    \"output_dir\": \"${PROFILE_DIR}\",
    \"start_step\": ${PROFILE_START_STEP},
    \"num_steps\": ${PROFILE_NUM_STEPS},
    "activities": ["CPU", "GPU"]
  }"

