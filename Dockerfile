FROM public.ecr.aws/k1t8c0v2/rayproject/ray-ml:2.6.3

USER root

RUN mkdir -p /home/model_weights

COPY download_model.py /download_model.py

RUN python /download_model.py

