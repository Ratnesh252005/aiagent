"""
Setup script for RAG Document Assistant
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Successfully installed all requirements")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            print("Creating .env file from template...")
            with open('.env.example', 'r') as example:
                content = example.read()
            
            with open('.env', 'w') as env_file:
                env_file.write(content)
            
            print("✅ Created .env file")
            print("⚠️  Please edit .env file and add your API keys before running the application")
            return True
        else:
            print("❌ .env.example file not found")
            return False
    else:
        print("✅ .env file already exists")
        return True

def main():
    """Main setup function"""
    print("🚀 Setting up RAG Document Assistant...")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return False
    
    # Create .env file
    if not create_env_file():
        print("❌ Setup failed during .env file creation")
        return False
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit the .env file and add your API keys:")
    print("   - PINECONE_API_KEY: Get from https://app.pinecone.io/")
    print("   - PINECONE_ENVIRONMENT: Your Pinecone environment")
    print("   - GEMINI_API_KEY: Get from https://makersuite.google.com/app/apikey")
    print("\n2. Run the application:")
    print("   streamlit run app.py")
    print("\n3. Open your browser and go to the displayed URL")
    
    return True

if __name__ == "__main__":
    main()
