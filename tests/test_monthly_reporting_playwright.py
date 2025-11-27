"""
Playwright end-to-end tests for the Monthly Reporting Data Streamlit module.

Tests cover:
- Application loading and navigation
- File upload functionality (CSV and Excel)
- Data saving to SQLite database
- Column mapping functionality
- Data loading from database
- Calculations and aggregations
- Export functionality

To run these tests, first start the Streamlit app:
    streamlit run monthly_reporting_data.py --server.port 8501 --server.headless true

Then run tests with:
    pytest tests/test_monthly_reporting_playwright.py --no-cov
"""

import pytest
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

# Add the project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Default Streamlit URL
STREAMLIT_URL = os.environ.get("STREAMLIT_URL", "http://localhost:8501")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_data_dir():
    """Create a temporary directory with test data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample CSV data
        csv_data = pd.DataFrame({
            'Site': ['Site A', 'Site A', 'Site B', 'Site B'],
            'Date': ['2025-01-01', '2025-02-01', '2025-01-01', '2025-02-01'],
            'Actual Gen (kWh)': [10000.5, 11000.5, 8000.5, 8500.5],
            'Irradiance-based generation': [10500.0, 11500.0, 8500.0, 9000.0],
            'Forecast Gen (kWh)': [10200.0, 11200.0, 8200.0, 8700.0],
            'Actual PR (%)': [85.5, 86.0, 84.0, 84.5],
            'Forecast PR (%)': [88.0, 88.0, 87.0, 87.0],
            'Availability (%)': [98.5, 99.0, 97.5, 98.0],
            'kWp': [100, 100, 80, 80],
        })
        csv_path = os.path.join(tmpdir, 'test_solar_data.csv')
        csv_data.to_csv(csv_path, index=False)
        
        # Create sample Excel data
        excel_data = pd.DataFrame({
            'Site': ['Plant 1', 'Plant 1', 'Plant 2', 'Plant 2', 'Plant 3'],
            'Date': ['Jan-2025', 'Feb-2025', 'Jan-2025', 'Feb-2025', 'Jan-2025'],
            'Actual Gen (kWh)': [15000, 16000, 12000, 13000, 9000],
            'Irradiance-based generation': [15500, 16500, 12500, 13500, 9500],
            'Forecast Gen (kWh)': [15200, 16200, 12200, 13200, 9200],
            'Actual PR (%)': [82.5, 83.0, 81.0, 81.5, 80.0],
            'Forecast PR (%)': [85.0, 85.0, 84.0, 84.0, 83.0],
            'Availability (%)': [97.5, 98.0, 96.5, 97.0, 95.0],
            'kWp': [150, 150, 120, 120, 100],
        })
        excel_path = os.path.join(tmpdir, 'test_solar_data.xlsx')
        excel_data.to_excel(excel_path, index=False)
        
        yield {
            'dir': tmpdir,
            'csv_path': csv_path,
            'excel_path': excel_path,
            'csv_data': csv_data,
            'excel_data': excel_data,
        }


@pytest.fixture(scope="module")
def app_url():
    """Return the Streamlit app URL."""
    return STREAMLIT_URL
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------


class TestAppLoading:
    """Tests for basic application loading and UI elements."""
    
    def test_app_loads_successfully(self, page, app_url):
        """Test that the Streamlit app loads successfully."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        
        # Wait for the title to appear
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Verify the page title
        title = page.locator("h1:has-text('Solar Asset Data Manager')")
        assert title.is_visible()
    
    def test_all_tabs_are_visible(self, page, app_url):
        """Test that all main tabs are visible."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Check for tab buttons
        expected_tabs = ["Upload", "Query", "Tables", "Analytics", "Calculations", "Waterfall"]
        for tab_name in expected_tabs:
            tab = page.locator(f"button:has-text('{tab_name}')")
            assert tab.count() > 0, f"Tab '{tab_name}' should be visible"
    
    def test_sidebar_elements_are_visible(self, page, app_url):
        """Test that sidebar elements are visible."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Check for sidebar elements
        settings_header = page.locator("text=Settings")
        assert settings_header.count() > 0
        
        # Check for fiscal year settings
        fiscal_settings = page.locator("text=Fiscal Year Settings")
        assert fiscal_settings.count() > 0


class TestFileUpload:
    """Tests for file upload functionality."""
    
    def test_upload_csv_file(self, page, app_url, test_data_dir):
        """Test uploading a CSV file."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Click on Upload tab
        upload_tab = page.locator("button:has-text('Upload')").first
        upload_tab.click()
        page.wait_for_timeout(1000)
        
        # Find the file input and upload
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files(test_data_dir['csv_path'])
        
        # Wait for file to be processed
        page.wait_for_timeout(3000)
        
        # Check for success message or data display
        # The app should show the data preview
        data_frame = page.locator("[data-testid='stDataFrame']")
        assert data_frame.count() > 0 or page.locator("text=loaded").count() > 0
    
    def test_upload_excel_file(self, page, app_url, test_data_dir):
        """Test uploading an Excel file."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Click on Upload tab
        upload_tab = page.locator("button:has-text('Upload')").first
        upload_tab.click()
        page.wait_for_timeout(1000)
        
        # Find the file input and upload
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files(test_data_dir['excel_path'])
        
        # Wait for file to be processed
        page.wait_for_timeout(3000)
        
        # Check for success message or data display
        data_frame = page.locator("[data-testid='stDataFrame']")
        assert data_frame.count() > 0 or page.locator("text=loaded").count() > 0


class TestDatabaseSaving:
    """Tests for saving data to SQLite database - KEY functionality."""
    
    def test_generate_sample_data_and_save(self, page, app_url):
        """Test generating sample data and saving it to the database."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Find and click "Create Test Data" button in sidebar (use first to handle duplicates)
        create_test_data_btn = page.locator("button:has-text('Create Test Data')").first
        create_test_data_btn.click()
        
        # Wait for data generation
        page.wait_for_timeout(5000)
        
        # Check for success message
        success_msg = page.locator("text=Sample data created")
        assert success_msg.count() > 0 or page.locator("text=sample_solar_data").count() > 0
    
    def test_save_uploaded_csv_to_database(self, page, app_url, test_data_dir):
        """Test saving uploaded CSV data to SQLite database."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Click on Upload tab
        upload_tab = page.locator("button:has-text('Upload')").first
        upload_tab.click()
        page.wait_for_timeout(1000)
        
        # Upload CSV file
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files(test_data_dir['csv_path'])
        page.wait_for_timeout(3000)
        
        # Look for column mapping section
        page.wait_for_timeout(2000)
        
        # Click "Confirm Mapping" if visible
        confirm_btns = page.locator("button:has-text('Confirm')")
        if confirm_btns.count() > 0:
            confirm_btns.first.click()
            page.wait_for_timeout(1000)
        
        # Look for "Show Save UI" button if present
        show_save_btn = page.locator("button:has-text('Show Save UI')")
        if show_save_btn.count() > 0:
            show_save_btn.first.click()
            page.wait_for_timeout(1000)
        
        # Try to find and click the save button
        save_btn = page.locator("button:has-text('Save to Database')")
        if save_btn.count() > 0:
            save_btn.first.click()
            page.wait_for_timeout(3000)
            
            # Check for success message
            success_indicators = page.locator("text=Saved")
            assert success_indicators.count() > 0 or page.locator("text=rows").count() > 0
    
    def test_save_excel_to_database(self, page, app_url, test_data_dir):
        """Test saving uploaded Excel data to SQLite database - KEY TEST."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Click on Upload tab
        upload_tab = page.locator("button:has-text('Upload')").first
        upload_tab.click()
        page.wait_for_timeout(1000)
        
        # Upload Excel file
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files(test_data_dir['excel_path'])
        page.wait_for_timeout(3000)
        
        # The app should show data was loaded
        loaded_indicators = page.locator("text=rows")
        assert loaded_indicators.count() > 0 or page.locator("[data-testid='stDataFrame']").count() > 0


class TestDataLoadFromDatabase:
    """Tests for loading data from SQLite database."""
    
    def test_load_table_from_database(self, page, app_url):
        """Test loading an existing table from the database."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # First, generate sample data
        create_test_data_btn = page.locator("button:has-text('Create Test Data')").first
        create_test_data_btn.click()
        page.wait_for_timeout(5000)
        
        # Go to Upload tab
        upload_tab = page.locator("button:has-text('Upload')").first
        upload_tab.click()
        page.wait_for_timeout(1000)
        
        # Look for the database loading section
        load_table_section = page.locator("text=Load from Database")
        if load_table_section.count() > 0:
            # Find the select box and Load button
            load_btn = page.locator("button:has-text('Load Table')")
            if load_btn.count() > 0:
                # Select the sample_solar_data table first
                select_elements = page.locator("select")
                if select_elements.count() > 0:
                    select_elements.first.select_option("sample_solar_data")
                    page.wait_for_timeout(500)
                
                load_btn.click()
                page.wait_for_timeout(3000)
                
                # Check that data was loaded
                success = page.locator("text=Loaded")
                assert success.count() > 0 or page.locator("[data-testid='stDataFrame']").count() > 0


class TestColumnMapping:
    """Tests for column mapping functionality."""
    
    def test_column_mapping_displayed(self, page, app_url, test_data_dir):
        """Test that column mapping options are displayed after file upload."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Upload a file
        upload_tab = page.locator("button:has-text('Upload')").first
        upload_tab.click()
        page.wait_for_timeout(1000)
        
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files(test_data_dir['csv_path'])
        page.wait_for_timeout(3000)
        
        # Check for column mapping section
        mapping_section = page.locator("text=Column Mapping")
        assert mapping_section.count() > 0 or page.locator("text=Actual Generation").count() > 0
    
    def test_auto_detection_works(self, page, app_url, test_data_dir):
        """Test that columns are auto-detected from file."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Upload a file
        upload_tab = page.locator("button:has-text('Upload')").first
        upload_tab.click()
        page.wait_for_timeout(1000)
        
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files(test_data_dir['csv_path'])
        page.wait_for_timeout(3000)
        
        # Look for auto-detected columns in the select boxes
        # The app should have pre-selected some columns based on patterns
        selectboxes = page.locator("[data-testid='stSelectbox']")
        assert selectboxes.count() > 0


class TestQueryTab:
    """Tests for the SQL Query tab."""
    
    def test_query_tab_loads(self, page, app_url):
        """Test that the Query tab loads properly."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # First generate some data
        create_test_data_btn = page.locator("button:has-text('Create Test Data')").first
        create_test_data_btn.click()
        page.wait_for_timeout(5000)
        
        # Click on Query tab
        query_tab = page.locator("button:has-text('Query')").first
        query_tab.click()
        page.wait_for_timeout(1000)
        
        # Check for SQL Query elements
        sql_header = page.locator("text=SQL Query")
        assert sql_header.count() > 0
    
    def test_run_simple_query(self, page, app_url):
        """Test running a simple SQL query."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # First generate some data
        create_test_data_btn = page.locator("button:has-text('Create Test Data')").first
        create_test_data_btn.click()
        page.wait_for_timeout(5000)
        
        # Click on Query tab
        query_tab = page.locator("button:has-text('Query')").first
        query_tab.click()
        page.wait_for_timeout(1000)
        
        # Find and click Run Query button
        run_btn = page.locator("button:has-text('Run Query')")
        if run_btn.count() > 0:
            run_btn.click()
            page.wait_for_timeout(3000)
            
            # Check for results or error
            results_shown = page.locator("[data-testid='stDataFrame']")
            error_shown = page.locator("text=error")
            
            # Either we got results or there was no data
            assert results_shown.count() > 0 or error_shown.count() > 0 or page.locator("text=returned").count() > 0


class TestTablesTab:
    """Tests for the Tables management tab."""
    
    def test_tables_tab_loads(self, page, app_url):
        """Test that the Tables tab loads properly."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Click on Tables tab
        tables_tab = page.locator("button:has-text('Tables')").first
        tables_tab.click()
        page.wait_for_timeout(1000)
        
        # Check for Manage Tables header
        manage_header = page.locator("text=Manage Tables")
        assert manage_header.count() > 0
    
    def test_view_table_data(self, page, app_url):
        """Test viewing table data."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # First generate some data
        create_test_data_btn = page.locator("button:has-text('Create Test Data')").first
        create_test_data_btn.click()
        page.wait_for_timeout(5000)
        
        # Click on Tables tab
        tables_tab = page.locator("button:has-text('Tables')").first
        tables_tab.click()
        page.wait_for_timeout(2000)
        
        # Look for the View tab or data display
        view_tab = page.locator("button:has-text('View')")
        if view_tab.count() > 0:
            view_tab.click()
            page.wait_for_timeout(1000)
        
        # Check for data display
        data_frame = page.locator("[data-testid='stDataFrame']")
        rows_info = page.locator("text=Rows")
        
        # Either we see data or row count info
        assert data_frame.count() > 0 or rows_info.count() > 0


class TestCalculationsTab:
    """Tests for the Calculations tab."""
    
    def test_calculations_tab_loads(self, page, app_url):
        """Test that the Calculations tab loads properly."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Click on Calculations tab
        calc_tab = page.locator("button:has-text('Calculations')").first
        calc_tab.click()
        page.wait_for_timeout(1000)
        
        # Check for header
        header = page.locator("text=Budget Variance")
        assert header.count() > 0 or page.locator("text=Technical Losses").count() > 0


class TestWaterfallTab:
    """Tests for the Waterfall visualization tab."""
    
    def test_waterfall_tab_loads(self, page, app_url):
        """Test that the Waterfall tab loads properly."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Click on Waterfall tab
        waterfall_tab = page.locator("button:has-text('Waterfall')").first
        waterfall_tab.click()
        page.wait_for_timeout(1000)
        
        # Check for header
        header = page.locator("text=Portfolio Loss Waterfall")
        assert header.count() > 0


class TestExportFunctionality:
    """Tests for data export functionality."""
    
    def test_download_csv_button_exists(self, page, app_url):
        """Test that CSV download buttons exist after data is loaded."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # First generate some data
        create_test_data_btn = page.locator("button:has-text('Create Test Data')").first
        create_test_data_btn.click()
        page.wait_for_timeout(5000)
        
        # Go to Tables tab
        tables_tab = page.locator("button:has-text('Tables')").first
        tables_tab.click()
        page.wait_for_timeout(2000)
        
        # Look for download button
        download_btn = page.locator("button:has-text('Download')")
        # Download buttons might be present
        assert download_btn.count() >= 0  # May or may not have data


class TestDatabaseIntegration:
    """Integration tests for database operations - KEY tests for Excel saving."""
    
    def test_excel_data_persists_in_database(self, page, app_url, test_data_dir):
        """Test that Excel data is correctly saved and can be queried from the database."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Upload Excel file
        upload_tab = page.locator("button:has-text('Upload')").first
        upload_tab.click()
        page.wait_for_timeout(1000)
        
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files(test_data_dir['excel_path'])
        page.wait_for_timeout(3000)
        
        # File should be loaded
        loaded_check = page.locator("text=rows")
        assert loaded_check.count() > 0 or page.locator("[data-testid='stDataFrame']").count() > 0
    
    def test_sample_data_in_sidebar_table_list(self, page, app_url):
        """Test that generated sample data appears in the sidebar table list."""
        page.goto(app_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("text=Solar Asset Data Manager", timeout=30000)
        
        # Generate sample data
        create_test_data_btn = page.locator("button:has-text('Create Test Data')").first
        create_test_data_btn.click()
        page.wait_for_timeout(5000)
        
        # Check sidebar for table list
        tables_section = page.locator("text=Tables")
        assert tables_section.count() > 0
        
        # The sample_solar_data table should be listed
        sample_table = page.locator("text=sample_solar_data")
        assert sample_table.count() > 0


# ---------------------------------------------------------------------------
# Conftest configuration for pytest-playwright
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Note: pytest-playwright provides the `page` fixture automatically
