// Custom JavaScript for Jupyter Notebook

// Function to add custom CSS
function add_custom_css() {
    var link = document.createElement('link');
    link.type = 'text/css';
    link.rel = 'stylesheet';
    link.href = require.toUrl('./utils/custom.css');
    $('head').append(link);
}

// Run when the notebook is fully loaded
function initialize_customizations() {
    // Add our custom CSS
    add_custom_css();
    
    // Fix markdown cells that might have been converted incorrectly
    $('.markdown-text').each(function() {
        let content = $(this).html();
        // Fix markdown headers that might be too large
        content = content.replace(/<h1[^>]*>(.*?)<\/h1>/g, '<h2>$1</h2>');
        content = content.replace(/<h2[^>]*>(.*?)<\/h2>/g, '<h3>$1</h3>');
        content = content.replace(/<h3[^>]*>(.*?)<\/h3>/g, '<h4>$1</h4>');
        $(this).html(content);
    });
}

// Run our initialization when the notebook is ready
$(document).ready(function() {
    // Check if we're in a notebook
    if (window.Jupyter && Jupyter.notebook) {
        // Run immediately
        initialize_customizations();
        
        // Also run after the notebook is fully loaded
        Jupyter.notebook.events.on('notebook_loaded.Notebook', initialize_customizations);
    }
});
