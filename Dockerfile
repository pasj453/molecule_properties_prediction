FROM conda/miniconda3:latest

WORKDIR app
COPY env.yml requirements.txt setup.py ./
RUN mkdir src models data

COPY src/ ./src

RUN conda update -n base conda -y \
	&& conda env create -f env.yml

ENV PATH /usr/local/envs/servier/bin:$PATH

RUN /bin/bash -c "source activate servier && conda install -y -c conda-forge rdkit && pip install ."

EXPOSE 5000

ENTRYPOINT ["servier"]
