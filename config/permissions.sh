DIR=/pfs/work9/workspace/scratch/ds_wi22230-LLM_GAN_MarketResearch

for user in ds_wi22181 ds_wi22094 ds_wi22230 ds_wi22010 ds_wi22219; do
    setfacl -Rm u:$user:rwX,d:u:$user:rwX "$DIR"
done
