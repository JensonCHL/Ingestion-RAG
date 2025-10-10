# RAG Application Docker Deployment

## Prerequisites
- Docker and Docker Compose installed
- Your configured `.env` file

## Installing Docker Compose on Ubuntu (Docker already installed)

Since you already have Docker running on your VM, you only need to install Docker Compose:

1. **Install Docker Compose:**
   ```bash
   sudo apt update
   sudo apt install docker-compose
   ```

2. **Verify installation:**
   ```bash
   docker --version
   docker-compose --version
   ```

3. **If the above doesn't work, install Docker Compose manually:**
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

4. **Verify the manual installation:**
   ```bash
   docker-compose --version
   ```

## Quick Start (Development)

1. Clone or copy your application files to the server
2. Ensure your `.env` file is in the root directory
3. Build and run the application:
   ```bash
   docker-compose up --build
   ```
4. Access the application at `http://localhost:8501`

## Production Deployment with Port 80

1. Clone or copy your application files to the server
2. Ensure your `.env` file is in the root directory
3. Build and run the production stack:
   ```bash
   docker-compose -f docker-compose.prod.yml up --build -d
   ```
4. Access the application at `http://YOUR_SERVER_IP` (port 80)

## Production Deployment (Raw HTTP - No Domain)

1. Clone or copy your application files to the server
2. Ensure your `.env` file is in the root directory
3. Build and run the production stack:
   ```bash
   docker-compose -f docker-compose.prod.yml up --build -d
   ```
4. Find your server's public IP address
5. Access the application at `http://YOUR_SERVER_IP_ADDRESS`

## GitHub Deployment

To deploy directly from GitHub:

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git push -u origin main
   ```

2. **On Your Server:**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   
   # Create your .env file with your configurations
   # (Only this file is ignored by git for security)
   
   # Run the application
   docker-compose up --build -d
   ```

3. **Updating Your Application:**
   ```bash
   # On your development machine, after making changes:
   git add .
   git commit -m "Description of changes"
   git push origin main
   
   # On your server:
   git pull origin main
   docker-compose up --build -d
   ```

## Making Your Application Publicly Accessible

To access your application via public IP:

1. **Find Your Server's Public IP:**
   - On Linux/Mac: `curl ifconfig.me` or `curl ipinfo.io/ip`
   - On Windows: Visit https://www.whatismyip.com/

2. **Configure Firewall:**
   - Allow inbound traffic on port 80 (HTTP)
   - On Ubuntu: `sudo ufw allow 80`
   - On CentOS/RHEL: `sudo firewall-cmd --permanent --add-port=80/tcp && sudo firewall-cmd --reload`
   - On Windows: Configure Windows Defender Firewall to allow port 80

3. **Router Configuration (If Behind NAT):**
   - Log into your router's admin panel
   - Set up port forwarding for port 80 to your server's local IP
   - Forward external port 80 to internal port 80 of your server

4. **Access Your Application:**
   - Open a browser on any device
   - Navigate to `http://YOUR_PUBLIC_IP_ADDRESS`

## Environment Variables

Make sure your `.env` file includes all necessary configurations:
- DEKA API keys and endpoints
- Qdrant connection details
- Supabase credentials
- Authentication credentials

## Data Persistence

Uploaded files and artifacts are persisted through Docker volumes:
- `uploads/` directory for PDF files
- `artifacts/` directory for any generated artifacts

## Updating the Application

To update the application:
1. Pull the latest code or copy updated files
2. Rebuild and restart containers:
   ```bash
   docker-compose -f docker-compose.prod.yml down
   docker-compose -f docker-compose.prod.yml up --build -d
   ```

## Troubleshooting

- Check container logs: `docker-compose logs`
- Ensure all required ports are open (80, 443 for production)
- Verify environment variables are correctly set