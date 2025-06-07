# AIEDU Cloud Deployment

## Overview

This is the **cloud deployment model** of the AIEDU RAG system, designed for high-performance, scalable production deployment using Google Cloud AI services and Vercel serverless architecture.

## Features

- **High Performance**: <2 second response times
- **Global Scalability**: Worldwide CDN distribution via Vercel
- **Auto-Scaling**: Serverless architecture scales automatically
- **Cost-Effective**: Pay-per-use with generous free tiers
- **Zero DevOps**: No server management required

## Technology Stack

- **LLM**: Google Gemini 2.5 Flash
- **STT**: Google Cloud Speech-to-Text API
- **TTS**: Google Cloud Text-to-Speech API
- **Vector Store**: Pinecone
- **Embeddings**: Google Vertex AI / OpenAI
- **Frontend**: Next.js + React
- **Deployment**: Vercel serverless functions

## Prerequisites

### Required Accounts
- **Google Cloud Platform**: With billing enabled
- **Vercel Account**: Pro plan recommended ($20/month)
- **Pinecone Account**: Starter plan ($70/month)
- **Development Environment**: Node.js 18+, Python 3.9+

### API Quotas (Free Tier)
- **Gemini 2.5 Flash**: 15 RPM, 1M tokens/month
- **Speech-to-Text**: 60 minutes/month
- **Text-to-Speech**: 4M characters/month
- **Pinecone**: 1 index, 100K vectors (trial)

## Google Cloud Setup

### 1. Create Project
```bash
# Install Google Cloud CLI
curl https://sdk.cloud.google.com | bash

# Create project
gcloud projects create aiedu-rag-system
gcloud config set project aiedu-rag-system
```

### 2. Enable APIs
```bash
gcloud services enable generativeai.googleapis.com
gcloud services enable speech.googleapis.com
gcloud services enable texttospeech.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

### 3. Create Service Account
```bash
gcloud iam service-accounts create aiedu-service-account \
    --display-name="AIEDU RAG System Service Account"

gcloud iam service-accounts keys create credentials.json \
    --iam-account=aiedu-service-account@aiedu-rag-system.iam.gserviceaccount.com
```

## Local Development Setup

### 1. Install Dependencies
```bash
cd AIEDU/cloud_deployment

# Install frontend dependencies
cd frontend && npm install

# Install Python dependencies
pip install -r requirements_cloud.txt
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env.local

# Edit .env.local with your credentials:
# GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
# GEMINI_API_KEY=your_gemini_api_key
# PINECONE_API_KEY=your_pinecone_api_key
# PINECONE_ENVIRONMENT=your_pinecone_environment
```

### 3. Run Development Server
```bash
# Install Vercel CLI
npm install -g vercel

# Start development server
vercel dev
```

## Production Deployment

### 1. Vercel Setup
```bash
# Login to Vercel
vercel login

# Deploy
vercel --prod
```

### 2. Environment Variables
Configure in Vercel dashboard:
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GEMINI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_ENVIRONMENT`
- `DEPLOYMENT_MODE=cloud`

## Expected Performance

### Response Times
- **LLM Inference**: <2 seconds (vs 3-5s local)
- **Speech Recognition**: <1 second
- **Text-to-Speech**: <1 second
- **Vector Search**: <500ms

### Scalability
- **Concurrent Users**: Unlimited (auto-scaling)
- **Global Latency**: <100ms via CDN
- **Uptime**: 99.9% SLA

## Cost Analysis

### Development (Free Tier)
- **Google Cloud AI**: $0 (within limits)
- **Vercel**: $0 (hobby plan)
- **Pinecone**: $0 (trial)
- **Total**: $0

### Production Usage
- **Google Cloud AI**: ~$10-30/month
- **Vercel Pro**: $20/month
- **Pinecone Starter**: $70/month
- **Total**: ~$100-120/month

## Monitoring & Optimization

### Cost Controls
```bash
# Set billing alerts
gcloud alpha billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="AIEDU Budget Alert" \
    --budget-amount=100USD
```

### Performance Monitoring
- Vercel Analytics (built-in)
- Google Cloud Monitoring
- Custom API usage tracking

## Migration from Local

A migration script will be provided to transfer:
- ChromaDB documents → Pinecone
- Configuration settings
- User data and preferences

## Troubleshooting

### Common Issues
1. **API Rate Limits**: Check quota usage in Google Cloud Console
2. **Vercel Deployment Fails**: Verify environment variables
3. **High Costs**: Monitor usage with billing alerts
4. **Slow Performance**: Check region configuration

### Support
- See `cursor_docs/` for detailed technical documentation
- Google Cloud Support (with paid plans)
- Vercel Support (with Pro plans)

## Security Best Practices

- Use service accounts with minimal permissions
- Rotate API keys regularly
- Monitor access logs
- Enable audit logging in Google Cloud
- Use Vercel's built-in security features

## Development Status

**Current Phase**: Architecture Foundation (Week 1)
**Next Phase**: Google Cloud Integration (Week 2-3)

This cloud deployment is under active development. The local deployment remains the stable option during development. 