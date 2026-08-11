# 容器
docker run  -it \
--name "zyf_521dpsk_fp8_0" \
--privileged \
--shm-size=200G \
--device=/dev/kfd \
--device=/dev/dri/ \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
--ulimit memlock=-1:-1 \
--ipc=host \
--network host \
--group-add video \
--privileged \
-v /usr/local/hyhal:/usr/local/hyhal:ro \
-v /opt/hyhal:/opt/hyhal:ro \
-v /data:/nvme:ro \
-v /parastor:/parastor:ro \
-v /module:/root/module \
-v /module1:/root/module1 \
-v /public/opendas/DL_DATA/llm-models:/root/llm-models:ro \
-v /public/home/zhangyf9:/public/home/zhangyf9 \
-v ~:/zhangyifan \
-v /public/home/zhangyf9/sglang:/root/sglang \
-v /public/home/zhangyf9/dtk:/root/dtk \
-v /public/home/zhangyf9/whl:/root/whl \
wangaq-deepseekv4:v2  /bin/bash
