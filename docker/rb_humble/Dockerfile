FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG USERNAME=seungjun
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=humble

RUN apt-get update && apt-get install -y \
    locales \
    curl \
    ca-certificates \
    gnupg2 \
    software-properties-common \
    sudo \
    git \
    build-essential \
    cmake \
    x11-apps \
    mesa-utils \
    libgl1-mesa-glx \
    && locale-gen en_US.UTF-8 \
    && add-apt-repository universe

RUN curl -sSL \
    https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-moveit \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-rmw-fastrtps-cpp \
    ros-humble-xacro \
    ros-humble-urdf-launch \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} \
       --gid ${USER_GID} \
       --create-home \
       --shell /bin/bash \
       ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" \
       > /etc/sudoers.d/${USERNAME}

RUN echo "source /opt/ros/humble/setup.bash" \
    >> /home/${USERNAME}/.bashrc

USER ${USERNAME}

WORKDIR /home/seungjun/isaac_ws

CMD ["/bin/bash"]
