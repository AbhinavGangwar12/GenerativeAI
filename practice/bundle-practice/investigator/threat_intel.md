# Global Threat Intelligence Database

## APT29 (Cozy Bear)
A highly sophisticated state-sponsored threat group. They frequently utilize spear-phishing to gain initial access. Once inside, they are known to drop custom PowerShell payloads into the `C:\Windows\Temp` directory to establish persistence and bypass standard endpoint detection before moving laterally.

## LockBit 3.0 Ransomware
A Ransomware-as-a-Service (RaaS) operation. After gaining initial access via compromised RDP credentials, LockBit disables local antivirus services. It then rapidly encrypts files over the SMB protocol, appending a randomized string to the file extensions, and drops a ransom note in every affected directory.