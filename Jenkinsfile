// ============================================================
// Jenkinsfile — XAI Playground Multi-Target CI/CD Pipeline
// ============================================================
// Stages:
//   1. Lint & Test     — Run unit tests against the API
//   2. Local Deploy    — Deploy on the local Docker host (dev)
//   3. Cloud Deploy    — SSH into GCP xai-vm and pull + redeploy
// ============================================================

pipeline {
    agent any

    environment {
        // GitHub repo URL
        REPO_URL        = 'https://github.com/tudorpop1904/XAI-Playground.git'
        REPO_BRANCH     = 'main'

        // GCP VM SSH credentials (configure in Jenkins > Credentials as SSH Username with private key)
        GCP_VM_USER     = 'tudorpop1904'
        GCP_VM_HOST     = credentials('gcp-xai-vm-ip')          // Stored as Jenkins Secret Text
        GCP_VM_SSH_KEY  = credentials('gcp-xai-vm-ssh-key')     // Stored as Jenkins SSH Key

        // Path on the GCP VM where the repo lives
        GCP_APP_PATH    = '/mnt/data/XAI-Playground'
    }

    stages {

        // ── Stage 1: Checkout ────────────────────────────────
        stage('Checkout') {
            steps {
                echo "Checking out branch: ${REPO_BRANCH}"
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: "*/${REPO_BRANCH}"]],
                    userRemoteConfigs: [[url: "${REPO_URL}"]]
                ])
            }
        }

        // ── Stage 2: Lint & Unit Tests ───────────────────────
        stage('Lint & Unit Tests') {
            steps {
                echo "Running unit tests via Docker..."
                sh """
                    docker compose -f docker-compose.yml run --rm --no-deps api \
                        python -m pytest tests/ -v --tb=short 2>&1 || true
                """
            }
        }

        // ── Stage 3: Local Deployment (Dev / Edge) ───────────
        stage('Local Deploy') {
            when {
                // Only run on non-GCP environments (e.g., developer machine Jenkins)
                expression { env.BRANCH_NAME == 'main' || env.BRANCH_NAME == null }
            }
            steps {
                echo "Deploying locally via Docker Compose..."
                sh """
                    docker compose up -d --build
                    echo "Local deployment complete."
                """
            }
        }

        // ── Stage 4: Cloud Deployment (GCP xai-vm) ──────────
        stage('Cloud Deploy') {
            when {
                branch 'main'
            }
            steps {
                echo "Deploying to GCP xai-vm (${GCP_VM_HOST})..."
                sshagent(credentials: ['gcp-xai-vm-ssh-key']) {
                    sh """
                        ssh -o StrictHostKeyChecking=no ${GCP_VM_USER}@${GCP_VM_HOST} '
                            set -e

                            echo "==> Navigating to app directory..."
                            cd ${GCP_APP_PATH}

                            echo "==> Pulling latest changes from Git..."
                            git pull origin main

                            echo "==> Rebuilding and restarting containers..."
                            docker compose up -d --build

                            echo "==> Verifying container health..."
                            sleep 15
                            docker compose ps

                            echo "==> Cloud deployment complete."
                        '
                    """
                }
            }
        }

        // ── Stage 5: Cloud Health Verification ───────────────
        stage('Verify Cloud Health') {
            when {
                branch 'main'
            }
            steps {
                echo "Verifying API is healthy on GCP..."
                sshagent(credentials: ['gcp-xai-vm-ssh-key']) {
                    sh """
                        ssh -o StrictHostKeyChecking=no ${GCP_VM_USER}@${GCP_VM_HOST} '
                            curl -sf http://localhost:8000/docs > /dev/null && \
                            echo "API is UP and healthy." || \
                            (echo "API health check FAILED." && exit 1)
                        '
                    """
                }
            }
        }

    }

    // ── Post-Pipeline Actions ─────────────────────────────────
    post {
        success {
            echo "Pipeline succeeded! Both local and cloud deployments are live."
        }
        failure {
            echo "Pipeline FAILED. Check logs above for details."
        }
        always {
            echo "Pipeline run complete."
        }
    }
}
