// Global configuration
const API_BASE = '/api';

// Utility functions
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.insertBefore(alertDiv, document.body.firstChild);
    
    setTimeout(() => alertDiv.remove(), 5000);
}

// Format numbers
function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

// Format percentage
function formatPercentage(num) {
    return (parseFloat(num) * 100).toFixed(2) + '%';
}

// Format currency
function formatCurrency(num) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(num);
}

// API Helper
async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showAlert(`Error: ${error.message}`, 'danger');
        throw error;
    }
}

// Chart helper
function createLineChart(elementId, title, xLabel, yLabel, data) {
    const trace = {
        y: data,
        mode: 'lines+markers',
        name: title,
        line: { color: '#007bff', width: 2 },
        marker: { size: 6 }
    };
    
    const layout = {
        title: title,
        xaxis: { title: xLabel },
        yaxis: { title: yLabel },
        hovermode: 'closest',
        plot_bgcolor: '#f8f9fa',
        paper_bgcolor: 'white'
    };
    
    Plotly.newPlot(elementId, [trace], layout);
}

// Local storage helpers
const Storage = {
    set: (key, value) => localStorage.setItem(key, JSON.stringify(value)),
    get: (key) => {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : null;
    },
    remove: (key) => localStorage.removeItem(key),
    clear: () => localStorage.clear()
};

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('Apex Black Box v4.0 - Ready');
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});

// Export data to CSV
function exportToCSV(filename, data) {
    let csv = '';
    
    if (Array.isArray(data) && data.length > 0) {
        // Headers
        const headers = Object.keys(data[0]);
        csv = headers.join(',') + '\n';
        
        // Data rows
        data.forEach(row => {
            const values = headers.map(header => {
                const value = row[header];
                return typeof value === 'string' && value.includes(',') ? `"${value}"` : value;
            });
            csv += values.join(',') + '\n';
        });
    }
    
    // Create blob and download
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}