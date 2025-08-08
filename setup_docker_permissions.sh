#!/bin/bash

echo "🔧 Setting up Docker permissions for RealSense camera..."

# Add current user to video and plugdev groups
echo "👤 Adding user to video and plugdev groups..."
sudo usermod -a -G video $USER
sudo usermod -a -G plugdev $USER

# Set proper permissions for video devices
echo "📹 Setting video device permissions..."
sudo chmod 666 /dev/video*

# Create udev rules for RealSense
echo "📋 Creating udev rules for RealSense..."
sudo tee /etc/udev/rules.d/99-realsense.rules > /dev/null <<EOF
# Intel RealSense D435i
SUBSYSTEM=="usb", ATTR{idVendor}=="8086", ATTR{idProduct}=="0b3a", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="8086", ATTR{idProduct}=="0b3a", ATTR{authorized}="1"

# Video devices
KERNEL=="video*", MODE="0666", GROUP="video"
EOF

# Reload udev rules
echo "🔄 Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

# Set USB device permissions
echo "🔌 Setting USB device permissions..."
sudo chmod 666 /dev/bus/usb/*/*

echo "✅ Docker permissions setup complete!"
echo "🔄 Please log out and log back in for group changes to take effect."
echo "🐳 You can now run the ORB-SLAM3 Docker container with camera access." 