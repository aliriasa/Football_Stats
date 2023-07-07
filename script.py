#python -m streamlit run script.py
import pandas as pd
import streamlit as st
import numpy as np


def main():
    # Load the dataframe
    st.set_page_config(
        page_title="Football Statistics",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    df = load_dataframe()  # Replace `load_dataframe()` with your code to load the dataframe

    # Add a sidebar for filtering (if needed)
    filtered_df = sidebar_filters(df)

    st.write(format_dataframe(filtered_df), unsafe_allow_html=True)


def load_dataframe():
    # Replace with your code to load the dataframe
    # For example:
    df = pd.read_csv('transfer.csv')
    return df

def sidebar_filters(df):
    # Add sidebar title
    st.sidebar.title('Data Filters')

    # Get the categorical column names
    categorical_columns = ['Liga', 'Equipo', 'Posicion', 'Pie', 'Lesionado']

    # Get the numerical column names
    numerical_columns = df.select_dtypes(include=[float, int]).columns.tolist()

    # Add filter options for categorical and numerical columns
    column_filters = st.sidebar.multiselect('Columns', categorical_columns + numerical_columns)

    # Initialize the filtered dataframe with the original dataframe
    filtered_df = df.copy()

    for column in column_filters:
        if column in categorical_columns:
            st.sidebar.markdown(f"**Filtering by {column}**")
            # Get unique values for the column, filtered based on previous filters
            column_values = filtered_df[column].loc[filtered_df[column].notnull()]
            unique_values = column_values.unique()
            # Display all possible values as a dropdown menu
            all_option = 'All' in column_values
            if all_option:
                selected_values = st.sidebar.multiselect(column, unique_values, default='All')
                if 'All' in selected_values:
                    filtered_df = filtered_df.drop(columns=[column])
            else:
                selected_values = st.sidebar.multiselect(column, unique_values)
                # Filter the dataframe based on the selected values
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
        else:
            # Apply the filters based on the selected numerical column
            st.sidebar.markdown(f"**Filtering by {column}**")
            min_value = st.sidebar.number_input(f'Minimum Value for {column}', value=filtered_df[column].min())
            max_value = st.sidebar.number_input(f'Maximum Value for {column}', value=filtered_df[column].max())
            filtered_df = filtered_df[(filtered_df[column] >= min_value) & (filtered_df[column] <= max_value)]

    # Return the filtered dataframe
    return filtered_df


def format_dataframe(df):
    # Create a copy of the dataframe to avoid modifying the original
    formatted_df = df.copy()

    # Define the columns to display initially
    columns_to_display = ['Liga', 'Equipo', 'Nombre', 'Imagen', 'Posicion', 'Nacionalidad', 'Valor de mercado']

    # Create an expander for displaying additional columns
    with st.expander("More Columns", expanded=False):
        # Get the list of remaining columns
        remaining_columns = [col for col in formatted_df.columns if col not in columns_to_display]

        # Add checkbox for each remaining column
        selected_columns = st.multiselect("Select columns to display", remaining_columns, default=[])
        columns_to_display.extend(selected_columns)

        # Add a dropdown menu to select a column for sorting
        sort_columns = st.multiselect("Select columns to sort", columns_to_display, default=['Valor de mercado'])

    # Filter the dataframe based on the selected columns
    formatted_df = formatted_df[columns_to_display]

    # Sort the dataframe based on the selected columns
    if sort_columns:
        # Sort the dataframe by multiple columns
        sort_orders = [False if pd.api.types.is_numeric_dtype(formatted_df[col].dtype) else True for col in sort_columns]

        # Convert NaN values to appropriate sentinel values for sorting
        sentinel = np.inf if sort_orders[0] else -np.inf
        formatted_df[sort_columns] = formatted_df[sort_columns].fillna(sentinel)

        # Perform multi-column sorting
        formatted_df = formatted_df.sort_values(by=sort_columns, ascending=sort_orders, na_position='last')

        # Restore NaN values in sorted columns
        formatted_df[sort_columns] = formatted_df[sort_columns].replace(sentinel, np.nan)


    # Replace the 'Imagen' column with formatted HTML to display the image
    formatted_df['Imagen'] = formatted_df['Imagen'].apply(lambda url: f'<img src="{url}" alt="Image" width="50px">')

    html_table = formatted_df.to_html(escape=False, index=False)

    return html_table



if __name__ == '__main__':
    main()


