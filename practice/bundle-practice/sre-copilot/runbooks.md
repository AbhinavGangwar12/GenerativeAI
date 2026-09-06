# Infrastructure Runbooks

## Kubernetes: CrashLoopBackOff
When a pod enters a CrashLoopBackOff state, it means the container is repeatedly failing to start. 
1. First, check the pod logs using: `kubectl logs <pod-name> -n <namespace> --previous`.
2. Inspect the pod events for scheduling errors using: `kubectl describe pod <pod-name> -n <namespace>`.
3. Verify that the environment variables and secrets mapped to the deployment exist.

## Nagios: High CPU Utilization
This alert triggers when CPU usage exceeds 90% for more than 5 minutes.
1. SSH into the affected node and run `top` or `htop` to identify the runaway process.
2. If it is an application pod, consider scaling the deployment horizontally using `kubectl scale`.
3. If it is a system-level process, check the Ansible configuration playbooks for misconfigured cron jobs.