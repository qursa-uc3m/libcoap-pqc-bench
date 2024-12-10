# Network Emulation

Running Dockerized network emulation can simplify things relative to using VMs as in Seoane's paper.

## Pumba vs dockerized NetEm

Pumba allows:

- delay
- packet loss

NetEm allows:

- delay
- packet loss
- packet duplication
- packet corruption
- packet reordering
- bandwidth rate

See:

- [Network emulation for Docker containers](https://medium.com/hackernoon/network-emulation-for-docker-containers-f4d36b656cc3)
- [Are Containers Coupled with NetEm a Reliable Tool for Performance Study of Network Protocols?](https://ieeexplore.ieee.org/document/9020466)

## NetEm

NOTE: netem only adds delay to packets leaving the interface. If you want to simulate bi-directional delay, two instances of tc netem - one on each end - are required. (from [here](https://srtlab.github.io/srt-cookbook/how-to-articles/using-netem-to-emulate-networks.html)).

### NetEM on VM with KVM

See: [How to Install and Configure KVM on Debian 11 Bullseye Linux](https://linux.how2shout.com/how-to-install-and-configure-kvm-on-debian-11-bullseye-linux/)

First check if your CPU supports hardware virtualization

```bash
egrep -c '(vmx|svm)' /proc/cpuinfo
```

If the output is 0, then your CPU doesn't support hardware virtualization, and you must stop here.

Install the cpu-checker package as follows.

```bash
sudo apt install -y cpu-checker
```

Then run the kvm-ok command, and if KVM virtualization is enabled, you should get the following output.

```bash
kvm-ok
```

You should get the following output.

```bash
INFO: /dev/kvm exists
KVM acceleration can be used
```

Install KVM and related tools

```bash
sudo apt update
sudo apt install libvirt-daemon-system libvirt-clients bridge-utils
```

You may need to start and Enable libvirtd if not already running:

```bash
sudo systemctl start libvirtd
sudo systemctl enable libvirtd
```

Then restart libvirtd

```bash
sudo systemctl restart libvirtd
```

Verify the status of the default network managed by libvirt and start it if it's not active.

Use

```bash
virsh net-list --all
```

to list all networks known to libvirt.
Check if the "default" network is listed and whether it's active. If it's not active, start it

```bash
sudo virsh net-start default
sudo virsh net-autostart default
```

Note that you may need to restart your computer for the "default" network to be listed.

Then

```bash
sudo virsh net-list --all
```

Ensure "virbr0" exists

```bash
ip a show virbr0
```

In Debian 11 we can use virt-install to create a new VM with qemu-kvm. This looks easier at first and command-line friendly. But it doesn't loooks easy to make it work with Ubuntu 20.04. So we will use the qemu-system-x86_64 command instead in the Ubuntu 20.04 case.

#### On Debian 11

In this case we need to install the qemu-kvm and virtinst packages

```bash
sudo apt update
sudo apt install qemu-kvm virtinst
```

You can use virt-install to create a new VM. The following is a basic example script to create a VM. Adjust the parameters like --name, --memory, --disk, and --location as needed.

```bash
sudo virt-install \
  --name netem-vm \
  --memory 2048 \
  --vcpus 2 \
  --disk path=/var/lib/libvirt/images/netem-vm.img,size=10 \
  --os-type linux \
  --os-variant debian10 \
  --network bridge=virbr0 \
  --graphics none \
  --console pty,target_type=serial \
  --location 'http://deb.debian.org/debian/dists/bullseye/main/installer-amd64/' \
  --extra-args 'console=ttyS0,115200n8 serial'
```

*Note*: on installation I have called the user `netem-vm` and the password `netem-vm`.

You can later start the VM with

```bash
sudo virsh start netem-vm
```

and connect to it with

```bash
sudo virsh console netem-vm
```

The escape sequence to exit the console is `Ctrl+Shift+^` in Spanish keyboards.

#### On Ubuntu 20.04

In this case we need to install the uml-utilities package

```bash
sudo apt update
sudo apt install uml-utilities
```

Download the Ubuntu ISO:

```bash
wget -P ~/Downloads https://cdimage.debian.org/cdimage/archive/10.13.0/amd64/iso-cd/debian-10.13.0-amd64-netinst.iso
```

Create a Disk Image for the VM:

```bash
qemu-img create -f qcow2 /var/lib/libvirt/images/netem-vm.qcow2 10G
```

Create a TAP device and attach it to virbr0:

```bash
sudo tunctl -t tap0 -u `whoami`
sudo ip link set tap0 up
sudo brctl addif virbr0 tap0
This creates a TAP device named tap0, brings it up, and attaches it to the virbr0 bridge. Replace whoami with your username if necessary.
```

Start the VM with QEMU/KVM and attach it to the TAP device:

```bash
sudo qemu-system-x86_64 -enable-kvm \
  -m 2048 \
  -smp cpus=2 \
  -hda /var/lib/libvirt/images/netem-vm.qcow2 \
  -cdrom ~/Downloads/debian-10.13.0-amd64-netinst.iso \
  -net nic,model=virtio -net tap,ifname=tap0,script=no,downscript=no
```

#### Once installed

Inside the VM get the IP address of the VM

```bash
hostname -I
```

Now let's install NetEm on the VM

```bash
sudo apt update
sudo apt install iproute2
```

Now enable IP forwarding on the VM

```bash
sudo sysctl -w net.ipv4.ip_forward=1
sudo sh -c 'echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf'
```

And route the traffic from the VM to the host machine (assuming the host machine has IP address 192.168.0.22 and the server uses port 12345, and that the new interface of the VM is enp1s0)

```bash
sudo iptables -t nat -A PREROUTING -p tcp --dport 12345 -j DNAT --to-destination 192.168.0.22
sudo iptables -A FORWARD -p tcp -d 192.168.0.22 --dport 12345 -j ACCEPT
sudo iptables -t nat -A POSTROUTING -o enp1s0 -j MASQUERADE
```

Now we can run NetEm on the VM to process the traffic. For example, to add 100ms of delay to all packets leaving enp1s0 (check if this is the correct interface), run

```bash
sudo tc qdisc add dev enp1s0 root netem delay 100ms
```

This can be removed with

```bash
sudo tc qdisc del dev enp1s0 root
```

Now if the VM and the host server are on different subnets, we need to configure NAT and Port Forwarding in the host Configure the host's iptables to forward traffic destined for a specific port to the VM, and then from the VM back to the server on the host.

```bash
sudo iptables -t nat -A PREROUTING -p tcp -s <client_port> --dport <server_port> -j DNAT --to-destination <vm_ip>
sudo iptables -A FORWARD -p tcp -d <vm_ip> --dport <server_port> -j ACCEPT
```

Replace `<server_port>` with the server's listening port, `<client_port>` with the client's port, and `<vm_ip>` with the VM's IP address.

But this doesn't seem to work because of some collision with other iptables rules (perhaps Docker or libvirt). So we can use the following instead

```bash
sudo iptables -t nat -I PREROUTING 1 -p tcp -s 192.168.0.30 --dport 12345 -j DNAT --to-destination 192.168.122.156
sudo iptables -I FORWARD 1 -p tcp -d 192.168.122.156 --dport 12345 -j ACCEPT
```

this inserts the rules at the beginning of the chain instead of appending them at the end. Specifying the client IP address seems necessary to avoid loops.
