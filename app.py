# Core pkgs
import streamlit as st


from routes.plotting import show_plotting
from routes.modelling import show_modelling
from routes.eda import show_testing
from routes.about import show_about
from kmcnew import kmc


def main():
    """Enablers of Confidence - Modular Streamlit App"""
    
    # Title
    st.title("Enablers of Confidence")
    
    # Sidebar navigation
    activities = ["About","EDA", "Plot", "Model Building","Testing"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    
    # Display the selected page
    if choice == 'EDA':
        # show_eda()
        show_testing()
    elif choice == 'Plot':
        show_plotting()
    elif choice == 'Model Building':
        show_modelling()
    # elif choice == 'Testing':
    #     show_testing()
    elif choice == 'About':
        show_about()
    elif choice == 'Testing':
        kmc()


        
if __name__ == '__main__':
    main()