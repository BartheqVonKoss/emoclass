FROM archlinux:latest

WORKDIR /usr/src/app
RUN sed -i '1i Server = http://mirrors.cat.net/archlinux/$repo/os/$arch' /etc/pacman.d/mirrorlist
RUN pacman -Sy --noconfirm python git python-pip
COPY requirements.txt ./
RUN python -m venv env
RUN . env/bin/activate
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# run the gradio gui
CMD [ "python", "app.py" ]

