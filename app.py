# Core pkgs
import streamlit as st

# Import page modules
from pages.eda import show_eda
from pages.plotting import show_plotting
from pages.modeling import show_modeling
from pages.testing import show_testing
from pages.about import show_about

def main():
    """Enablers of Confidence - Modular Streamlit App"""
    
    # Title
    st.title("Enablers of Confidence")
    
    # Sidebar navigation
    activities = ["EDA", "Plot", "Model Building", "Testing", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    
    # Display the selected page
    if choice == 'EDA':
        show_eda()
    elif choice == 'Plot':
        show_plotting()
    elif choice == 'Model Building':
        show_modeling()
    elif choice == 'Testing':
        show_testing()
    elif choice == 'About':
        show_about()

if __name__ == '__main__':
    main()