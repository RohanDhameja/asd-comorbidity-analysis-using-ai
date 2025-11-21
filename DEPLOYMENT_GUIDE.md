# 🚀 Deployment Guide - ASD Comorbidity & SDOH Dashboard

## Table of Contents
1. [Local Deployment](#local-deployment)
2. [Streamlit Cloud (Free)](#streamlit-cloud-free)
3. [Heroku Deployment](#heroku-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Troubleshooting](#troubleshooting)

---

## Local Deployment

### Option 1: Using the Run Script (Easiest)

```bash
# Navigate to project directory
cd /Users/rhuria/IYRC

# Run the script
./run_dashboard.sh
```

### Option 2: Manual Launch

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py
```

The dashboard will open at: **http://localhost:8501**

---

## Streamlit Cloud (Free) ⭐ RECOMMENDED

Perfect for presentations and sharing your research!

### Prerequisites
- GitHub account
- Your code pushed to a GitHub repository

### Steps:

1. **Push Code to GitHub**
   ```bash
   cd /Users/rhuria/IYRC
   git init
   git add .
   git commit -m "Initial commit: ASD Comorbidity Dashboard"
   
   # Create a new repository on GitHub first, then:
   git remote add origin https://github.com/YOUR_USERNAME/asd-sdoh-dashboard.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repository
   - Set:
     - **Main file path**: `app.py`
     - **Python version**: 3.9 or higher
   - Click "Deploy"

3. **Your Dashboard is Live! 🎉**
   - You'll get a URL like: `https://your-app-name.streamlit.app`
   - Share this URL for your Stanford Conference presentation!

### Benefits:
- ✅ **Free** hosting
- ✅ **HTTPS** automatically
- ✅ Auto-updates when you push to GitHub
- ✅ Easy to share
- ✅ No server management

---

## Heroku Deployment

### Prerequisites
- Heroku account (free tier available)
- Heroku CLI installed

### Additional Files Needed:

1. **Create `Procfile`:**
   ```bash
   echo "web: sh setup.sh && streamlit run app.py" > Procfile
   ```

2. **Create `setup.sh`:**
   ```bash
   cat > setup.sh << 'EOF'
   mkdir -p ~/.streamlit/
   
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   EOF
   ```

3. **Create `runtime.txt`:**
   ```bash
   echo "python-3.9.16" > runtime.txt
   ```

### Deploy:

```bash
# Login to Heroku
heroku login

# Create app
heroku create asd-sdoh-dashboard

# Deploy
git push heroku main

# Open app
heroku open
```

---

## Docker Deployment

### Create Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run:

```bash
# Build image
docker build -t asd-dashboard .

# Run container
docker run -p 8501:8501 asd-dashboard
```

Access at: **http://localhost:8501**

### Deploy to Cloud:

#### AWS ECS / Google Cloud Run / Azure Container Instances
```bash
# Tag and push to registry
docker tag asd-dashboard:latest YOUR_REGISTRY/asd-dashboard:latest
docker push YOUR_REGISTRY/asd-dashboard:latest

# Deploy using cloud provider's CLI/Console
```

---

## Advanced: Custom Domain Setup

### For Streamlit Cloud:

1. Go to your app settings
2. Navigate to "Custom domain"
3. Add your domain (e.g., `asd-research.yourdomain.com`)
4. Update DNS records as instructed
5. Wait for DNS propagation (5-30 minutes)

### For Heroku:

```bash
heroku domains:add asd-research.yourdomain.com

# Add DNS record:
# CNAME: asd-research → your-heroku-app.herokuapp.com
```

---

## Sharing Your Dashboard

### For Presentations (Stanford Conference):

1. **Deploy to Streamlit Cloud** (easiest)
2. **Create a QR Code** for easy access:
   ```python
   import qrcode
   qr = qrcode.make('https://your-dashboard-url.streamlit.app')
   qr.save('dashboard_qr.png')
   ```
3. **Add QR code to your presentation slides**
4. **Test on mobile devices** before the conference

### For Research Publication:

Include in your paper:
```markdown
Interactive dashboard available at: https://your-dashboard-url.streamlit.app

Source code: https://github.com/YOUR_USERNAME/asd-sdoh-dashboard
```

---

## Performance Optimization

### For Large Datasets:

1. **Use caching** (already implemented with `@st.cache_data`)
2. **Downsample data** for visualization:
   ```python
   if len(data) > 10000:
       display_data = data.sample(n=10000, random_state=42)
   ```
3. **Load data lazily**
4. **Use pagination** for large tables

### Speed Up Loading:

- Pre-compute SHAP values and save as pickle
- Use lighter visualization libraries where possible
- Compress images
- Enable gzip compression

---

## Security Considerations

### For Public Deployment:

1. **Remove sensitive data** from CSV
2. **Add authentication** if needed:
   ```python
   import streamlit_authenticator as stauth
   ```
3. **Rate limiting** (Streamlit Cloud handles this)
4. **Environment variables** for any API keys:
   ```python
   import os
   api_key = os.getenv('API_KEY')
   ```

---

## Troubleshooting

### Issue: "Module not found" error

**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Dashboard loads slowly

**Solutions:**
- Check internet connection
- Clear Streamlit cache: `streamlit cache clear`
- Reduce dataset size for testing
- Check server resources

### Issue: SHAP plots not showing

**Solution:**
- Ensure Random Forest or XGBoost is selected
- Check that `shap` package is installed correctly
- Try: `pip install --upgrade shap`

### Issue: Deployment fails on Streamlit Cloud

**Common fixes:**
1. Check `requirements.txt` is in root directory
2. Verify Python version compatibility
3. Check file paths (use relative paths)
4. Review deployment logs in Streamlit Cloud dashboard

### Issue: Port already in use

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Issue: Cannot access from other devices

**Solution:**
```bash
# Allow external connections
streamlit run app.py --server.address 0.0.0.0
```

---

## Monitoring & Analytics

### Add Google Analytics (Optional):

1. Create `.streamlit/config.toml`:
   ```toml
   [browser]
   gatherUsageStats = true
   ```

2. Add tracking code to `app.py`:
   ```python
   st.markdown("""
   <!-- Google Analytics -->
   <script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
   """, unsafe_allow_html=True)
   ```

---

## Updating Your Dashboard

### Local Changes:

```bash
# Make changes to app.py
# Test locally
streamlit run app.py

# Commit and push
git add .
git commit -m "Update: Added new feature"
git push
```

### Streamlit Cloud auto-updates within minutes!

---

## Cost Comparison

| Platform | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| **Streamlit Cloud** | Yes (unlimited public apps) | $0/month | Research, presentations |
| **Heroku** | Yes (1 dyno, sleeps) | $7+/month | Small projects |
| **AWS/GCP/Azure** | Limited free tier | $10+/month | Production apps |
| **Localhost** | Free | Free | Development, testing |

---

## Next Steps

1. ✅ Deploy locally and test all features
2. ✅ Push code to GitHub
3. ✅ Deploy to Streamlit Cloud
4. ✅ Test on mobile devices
5. ✅ Share URL for Stanford Conference
6. ✅ Create backup (download deployed app data)
7. ✅ Set up monitoring (optional)

---

## Support & Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **Streamlit Community**: https://discuss.streamlit.io
- **SHAP Documentation**: https://shap.readthedocs.io
- **This Project's Issues**: Create issues on your GitHub repo

---

## For Your Stanford Conference

### Presentation Tips:

1. **Demo the dashboard live** (make sure wifi works!)
2. **Have backup screenshots** in case of connectivity issues
3. **Prepare QR code** for audience to access
4. **Highlight interactive features**:
   - Model comparison
   - SHAP analysis
   - Feature exploration
5. **Show data filtering** capabilities

### Backup Plan:

Record a screen capture video of the dashboard in action:
```bash
# macOS
# Use QuickTime Player → File → New Screen Recording

# Windows
# Use Windows Game Bar (Win + G)
```

---

**🎉 You're ready to deploy! Good luck with your Stanford Conference presentation!**

For questions: rohan.dhameja27@bcp.org


