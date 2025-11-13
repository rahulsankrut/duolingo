# Setting Up on Another Laptop

## Option 1: Clone Fresh (If you don't have the code yet)

```bash
# Clone the repository
git clone https://github.com/rahulsankrut/duolingo.git
cd duolingo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your configuration
# Copy the .env file from your other laptop or create a new one
nano .env  # or use your preferred editor
```

## Option 2: Pull Latest Changes (If you already have the code)

```bash
# Navigate to your project directory
cd /path/to/duolingo

# Make sure you're on the main branch
git checkout main

# Fetch the latest changes from GitHub
git fetch origin

# Pull and merge the latest changes
git pull origin main
```

## Option 3: If You Have Local Changes You Want to Keep

```bash
# Save your local changes (if any)
git stash

# Pull the latest changes
git pull origin main

# Restore your local changes (if needed)
git stash pop
```

## After Pulling/Cloning

1. **Activate virtual environment** (if not already active):
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install/Update dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `.env` from your other laptop, or
   - Create a new `.env` file with your Google Cloud credentials

4. **Verify setup**:
   ```bash
   # Check that all files are there
   ls -la backend/
   ls -la Frontend/
   
   # Test that Python can import modules
   python -c "from backend.config import GOOGLE_CLOUD_PROJECT; print('Config loaded')"
   ```

5. **Start the application**:
   ```bash
   # Terminal 1: Start backend
   cd backend
   python app.py
   
   # Terminal 2: Start frontend
   cd Frontend
   python3 -m http.server 8080
   ```

## Troubleshooting

### "Repository not found" error
- Make sure you're using the correct repository URL
- Verify you have access to the repository

### Merge conflicts
If you have local changes that conflict:
```bash
# See what files have conflicts
git status

# Resolve conflicts manually, then:
git add <resolved-files>
git commit -m "Resolve merge conflicts"
```

### Authentication issues
If you need to authenticate:
```bash
# Set up Google Cloud credentials
gcloud auth application-default login

# Verify your .env file has the correct project ID
cat .env
```

