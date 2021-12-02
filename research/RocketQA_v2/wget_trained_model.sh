wget -nv --no-check-certificate https://rocketqa.bj.bcebos.com/V2/checkpoint.tar.gz
tar -xzf checkpoint.tar.gz
cd checkpoint

for filename in marco_joint-encoder_trained.tar.gz marco_joint-encoder_warmup.tar.gz nq_joint-encoder_trained.tar.gz nq_joint-encoder_warmup.tar.gz;do
    tar -zxf ${filename}
    rm -rf ${filename}
done

cd ..
