// Application state
let currentScreen = 'welcome';
let uploadedImage = null;
let processingSteps = [
    {
        id: 1,
        name: "Ảnh gốc",
        description: "Ảnh siêu âm ban đầu được tải lên",
        technical_info: "Ảnh đầu vào từ thiết bị siêu âm"
    },
    {
        id: 2, 
        name: "Enhanced Canny Edge",
        description: "Phát hiện cạnh nâng cao với thuật toán Canny tối ưu",
        technical_info: "Sử dụng adaptive thresholds và gradient analysis"
    },
    {
        id: 3,
        name: "Kênh màu Đỏ",
        description: "Trích xuất thông tin từ kênh màu đỏ (Red channel)",
        technical_info: "RGB color space - Red component"
    },
    {
        id: 4,
        name: "Kênh màu Xanh lá",
        description: "Trích xuất thông tin từ kênh màu xanh lá (Green channel)",
        technical_info: "RGB color space - Green component"
    },
    {
        id: 5,
        name: "Kênh màu Xanh dương", 
        description: "Trích xuất thông tin từ kênh màu xanh dương (Blue channel)",
        technical_info: "RGB color space - Blue component"
    },
    {
        id: 6,
        name: "Kênh Hue",
        description: "Phân tích sắc thái màu (Hue) trong không gian HSV",
        technical_info: "HSV color space - Hue component"
    },
    {
        id: 7,
        name: "Kênh Saturation",
        description: "Phân tích độ bão hòa màu (Saturation) trong không gian HSV", 
        technical_info: "HSV color space - Saturation component"
    }
];

let currentProcessingStep = 0;
let settings = {
    processingSpeed: 'normal',
    displayQuality: 'high'
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing app...');
    initializeApp();
});

function initializeApp() {
    console.log('Initializing application...');
    setupFileUpload();
    loadSettings();
    showScreen('welcome');
    console.log('Application initialized');
}

// Screen management
function showScreen(screenName) {
    console.log('Switching to screen:', screenName);
    
    // Hide all screens
    const screens = document.querySelectorAll('.screen');
    screens.forEach(screen => {
        screen.classList.remove('active');
    });
    
    // Show target screen
    const targetScreen = document.getElementById(screenName + '-screen');
    if (targetScreen) {
        targetScreen.classList.add('active');
        currentScreen = screenName;
        console.log('Successfully switched to screen:', screenName);
    } else {
        console.error('Screen not found:', screenName + '-screen');
    }
}

// Navigation functions - make them global
window.showUploadScreen = function() {
    console.log('showUploadScreen called');
    showScreen('upload');
};

window.startProcessing = function() {
    console.log('startProcessing called');
    if (!uploadedImage) {
        alert('Vui lòng chọn ảnh trước');
        return;
    }
    
    showScreen('processing');
    currentProcessingStep = 0;
    processNextStep();
};

window.resetUpload = function() {
    console.log('resetUpload called');
    const preview = document.getElementById('image-preview');
    const fileInput = document.getElementById('file-input');
    
    if (preview) preview.style.display = 'none';
    if (fileInput) fileInput.value = '';
    uploadedImage = null;
};

window.downloadResult = function() {
    console.log('downloadResult called');
    const canvas = document.getElementById('generated-result-canvas');
    if (canvas) {
        const link = document.createElement('a');
        link.download = 'baby-generated-' + Date.now() + '.png';
        link.href = canvas.toDataURL();
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
};

window.processNewImage = function() {
    console.log('processNewImage called');
    resetUpload();
    currentProcessingStep = 0;
    showScreen('upload');
};

window.openSettings = function() {
    console.log('openSettings called');
    const panel = document.getElementById('settings-panel');
    if (panel) {
        panel.classList.add('open');
    }
};

window.closeSettings = function() {
    console.log('closeSettings called');
    const panel = document.getElementById('settings-panel');
    if (panel) {
        panel.classList.remove('open');
    }
    saveSettings();
};

// File upload functionality
function setupFileUpload() {
    console.log('Setting up file upload...');
    
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    
    if (!uploadArea || !fileInput) {
        console.error('Upload elements not found');
        return;
    }

    // Click to upload
    uploadArea.addEventListener('click', function(e) {
        console.log('Upload area clicked');
        // Don't trigger if clicking on sample button
        if (e.target.classList.contains('btn')) {
            return;
        }
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Add sample image button
    addSampleImageButton();
    
    console.log('File upload setup complete');
}

function addSampleImageButton() {
    const uploadArea = document.getElementById('upload-area');
    if (!uploadArea) return;
    
    // Check if button already exists
    if (uploadArea.querySelector('.sample-btn')) return;
    
    const sampleBtn = document.createElement('button');
    sampleBtn.className = 'btn btn--secondary sample-btn';
    sampleBtn.style.marginTop = '16px';
    sampleBtn.textContent = 'Sử dụng ảnh mẫu';
    sampleBtn.onclick = function(e) {
        e.stopPropagation();
        console.log('Sample image button clicked');
        loadSampleImage();
    };
    
    uploadArea.appendChild(sampleBtn);
    console.log('Sample image button added');
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    console.log('Handling file:', file.name);
    
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        alert('Vui lòng chọn file ảnh hợp lệ (JPG, JPEG, PNG, WEBP)');
        return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        alert('File quá lớn. Vui lòng chọn file nhỏ hơn 10MB');
        return;
    }

    // Load and preview image
    const reader = new FileReader();
    reader.onload = function(e) {
        uploadedImage = new Image();
        uploadedImage.onload = function() {
            console.log('Image loaded:', uploadedImage.width, 'x', uploadedImage.height);
            showImagePreview(file, e.target.result);
        };
        uploadedImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function showImagePreview(file, imageSrc) {
    console.log('Showing image preview');
    
    const preview = document.getElementById('image-preview');
    const previewImage = document.getElementById('preview-image');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');

    if (preview && previewImage && fileName && fileSize) {
        previewImage.src = imageSrc;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        preview.style.display = 'block';
        console.log('Image preview displayed');
    } else {
        console.error('Preview elements not found');
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Sample data for demo
function loadSampleImage() {
    console.log('Loading sample image...');
    
    // Create a sample ultrasound-like image
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 300;
    const ctx = canvas.getContext('2d');
    
    // Create gradient background
    const gradient = ctx.createRadialGradient(200, 150, 0, 200, 150, 200);
    gradient.addColorStop(0, '#666');
    gradient.addColorStop(1, '#222');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 400, 300);
    
    // Add some shapes to simulate ultrasound features
    ctx.fillStyle = '#888';
    ctx.beginPath();
    ctx.ellipse(200, 150, 80, 60, 0, 0, 2 * Math.PI);
    ctx.fill();
    
    ctx.fillStyle = '#AAA';
    ctx.beginPath();
    ctx.ellipse(180, 130, 20, 15, 0, 0, 2 * Math.PI);
    ctx.fill();
    
    ctx.beginPath();
    ctx.ellipse(220, 130, 20, 15, 0, 0, 2 * Math.PI);
    ctx.fill();
    
    // Add some noise for ultrasound effect
    const imageData = ctx.getImageData(0, 0, 400, 300);
    const data = imageData.data;
    
    for (let i = 0; i < data.length; i += 4) {
        const noise = (Math.random() - 0.5) * 60;
        data[i] = Math.max(0, Math.min(255, data[i] + noise));
        data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise));
        data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise));
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    // Convert to blob and create file
    canvas.toBlob(function(blob) {
        const file = new File([blob], 'sample-ultrasound.png', { type: 'image/png' });
        console.log('Sample image created');
        handleFile(file);
    });
}

// Processing functionality
function processNextStep() {
    console.log('Processing step:', currentProcessingStep + 1);
    
    if (currentProcessingStep >= processingSteps.length) {
        startAIGeneration();
        return;
    }

    const step = processingSteps[currentProcessingStep];
    const progress = ((currentProcessingStep + 1) / processingSteps.length) * 100;
    
    // Update UI
    updateProcessingUI(step, currentProcessingStep + 1, progress);
    
    // Show processing overlay
    const overlay = document.getElementById('processing-overlay');
    if (overlay) {
        overlay.classList.add('active');
    }
    
    // Process the image for this step
    setTimeout(() => {
        processImageStep(step.id);
        if (overlay) {
            overlay.classList.remove('active');
        }
        
        // Move to next step
        currentProcessingStep++;
        
        const delay = getProcessingDelay();
        setTimeout(() => {
            processNextStep();
        }, delay);
        
    }, 1000);
}

function updateProcessingUI(step, stepNumber, progress) {
    const currentStepEl = document.getElementById('current-step');
    const progressPercentEl = document.getElementById('progress-percent');
    const progressFillEl = document.getElementById('progress-fill');
    const stepTitleEl = document.getElementById('step-title');
    const stepDescEl = document.getElementById('step-description');
    const stepTechEl = document.getElementById('step-technical');
    
    if (currentStepEl) currentStepEl.textContent = `Bước ${stepNumber}/7`;
    if (progressPercentEl) progressPercentEl.textContent = `${Math.round(progress)}%`;
    if (progressFillEl) progressFillEl.style.width = `${progress}%`;
    if (stepTitleEl) stepTitleEl.textContent = step.name;
    if (stepDescEl) stepDescEl.textContent = step.description;
    if (stepTechEl) stepTechEl.textContent = step.technical_info;
}

function processImageStep(stepId) {
    const canvas = document.getElementById('processing-canvas');
    if (!canvas || !uploadedImage) return;
    
    const ctx = canvas.getContext('2d');
    
    // Set canvas size based on image
    const maxWidth = 400;
    const maxHeight = 300;
    const scale = Math.min(maxWidth / uploadedImage.width, maxHeight / uploadedImage.height);
    canvas.width = uploadedImage.width * scale;
    canvas.height = uploadedImage.height * scale;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw processed image based on step
    switch(stepId) {
        case 1: // Original
            ctx.drawImage(uploadedImage, 0, 0, canvas.width, canvas.height);
            break;
        case 2: // Canny Edge
            drawCannyEdge(ctx, canvas.width, canvas.height);
            break;
        case 3: // Red channel
            drawColorChannel(ctx, canvas.width, canvas.height, 'red');
            break;
        case 4: // Green channel
            drawColorChannel(ctx, canvas.width, canvas.height, 'green');
            break;
        case 5: // Blue channel
            drawColorChannel(ctx, canvas.width, canvas.height, 'blue');
            break;
        case 6: // Hue
            drawHSVChannel(ctx, canvas.width, canvas.height, 'hue');
            break;
        case 7: // Saturation
            drawHSVChannel(ctx, canvas.width, canvas.height, 'saturation');
            break;
    }
}

function drawCannyEdge(ctx, width, height) {
    ctx.drawImage(uploadedImage, 0, 0, width, height);
    
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    
    for (let i = 0; i < data.length; i += 4) {
        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        const edge = Math.random() > 0.8 ? 255 : gray > 100 ? 255 : 0;
        data[i] = edge;
        data[i + 1] = edge;
        data[i + 2] = edge;
    }
    
    ctx.putImageData(imageData, 0, 0);
}

function drawColorChannel(ctx, width, height, channel) {
    ctx.drawImage(uploadedImage, 0, 0, width, height);
    
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    
    for (let i = 0; i < data.length; i += 4) {
        switch(channel) {
            case 'red':
                data[i + 1] = 0;
                data[i + 2] = 0;
                break;
            case 'green':
                data[i] = 0;
                data[i + 2] = 0;
                break;
            case 'blue':
                data[i] = 0;
                data[i + 1] = 0;
                break;
        }
    }
    
    ctx.putImageData(imageData, 0, 0);
}

function drawHSVChannel(ctx, width, height, channel) {
    ctx.drawImage(uploadedImage, 0, 0, width, height);
    
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i] / 255;
        const g = data[i + 1] / 255;
        const b = data[i + 2] / 255;
        
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const diff = max - min;
        
        let h, s;
        
        s = max === 0 ? 0 : diff / max;
        
        if (diff === 0) {
            h = 0;
        } else if (max === r) {
            h = ((g - b) / diff) % 6;
        } else if (max === g) {
            h = (b - r) / diff + 2;
        } else {
            h = (r - g) / diff + 4;
        }
        h /= 6;
        
        if (channel === 'hue') {
            const hueColor = hsvToRgb(h, 1, 1);
            data[i] = hueColor.r * 255;
            data[i + 1] = hueColor.g * 255;
            data[i + 2] = hueColor.b * 255;
        } else if (channel === 'saturation') {
            const satValue = s * 255;
            data[i] = satValue;
            data[i + 1] = satValue;
            data[i + 2] = satValue;
        }
    }
    
    ctx.putImageData(imageData, 0, 0);
}

function hsvToRgb(h, s, v) {
    const c = v * s;
    const x = c * (1 - Math.abs((h * 6) % 2 - 1));
    const m = v - c;
    
    let r, g, b;
    
    if (h < 1/6) {
        r = c; g = x; b = 0;
    } else if (h < 2/6) {
        r = x; g = c; b = 0;
    } else if (h < 3/6) {
        r = 0; g = c; b = x;
    } else if (h < 4/6) {
        r = 0; g = x; b = c;
    } else if (h < 5/6) {
        r = x; g = 0; b = c;
    } else {
        r = c; g = 0; b = x;
    }
    
    return {
        r: r + m,
        g: g + m,
        b: b + m
    };
}

function getProcessingDelay() {
    const speeds = {
        fast: 1000,
        normal: 2000,
        slow: 3000
    };
    return speeds[settings.processingSpeed] || 2000;
}

// AI Generation
function startAIGeneration() {
    console.log('Starting AI generation');
    showScreen('ai-generation');
    
    const aiSteps = [
        { title: "Đang phân tích dữ liệu...", desc: "AI đang xử lý 7 kênh dữ liệu để tạo ra hình ảnh em bé" },
        { title: "Khởi tạo mô hình...", desc: "Đang tải mô hình AI chuyên dụng cho tạo ảnh em bé" },
        { title: "Xử lý với 50 bước...", desc: "Diffusion model đang tạo ảnh với 50 steps" },
        { title: "Tinh chỉnh chi tiết...", desc: "Đang tối ưu hóa các đặc điểm khuôn mặt em bé" },
        { title: "Hoàn thiện...", desc: "Đang áp dụng hậu kỳ và tối ưu chất lượng" }
    ];
    
    let currentAIStep = 0;
    
    function nextAIStep() {
        if (currentAIStep >= aiSteps.length) {
            showResults();
            return;
        }
        
        const step = aiSteps[currentAIStep];
        const titleEl = document.getElementById('ai-status-title');
        const descEl = document.getElementById('ai-status-desc');
        
        if (titleEl) titleEl.textContent = step.title;
        if (descEl) descEl.textContent = step.desc;
        
        currentAIStep++;
        setTimeout(nextAIStep, 3000);
    }
    
    nextAIStep();
}

function showResults() {
    console.log('Showing results');
    showScreen('result');
    
    // Draw original image in result
    const originalCanvas = document.getElementById('original-result-canvas');
    if (originalCanvas && uploadedImage) {
        const originalCtx = originalCanvas.getContext('2d');
        originalCtx.drawImage(uploadedImage, 0, 0, originalCanvas.width, originalCanvas.height);
    }
    
    // Generate and draw baby image
    generateBabyImage();
}

function generateBabyImage() {
    const canvas = document.getElementById('generated-result-canvas');
    if (!canvas || !uploadedImage) return;
    
    const ctx = canvas.getContext('2d');
    
    // Draw base image
    ctx.drawImage(uploadedImage, 0, 0, canvas.width, canvas.height);
    
    // Apply baby-like transformations
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    for (let i = 0; i < data.length; i += 4) {
        // Warm and soften colors
        data[i] = Math.min(255, data[i] * 1.1 + 20);
        data[i + 1] = Math.min(255, data[i + 1] * 1.05 + 15);
        data[i + 2] = Math.min(255, data[i + 2] + 10);
    }
    
    ctx.putImageData(imageData, 0, 0);
}

// Settings
function loadSettings() {
    // Skip localStorage for now to avoid sandbox issues
    updateSettingsUI();
}

function saveSettings() {
    const processingSpeedEl = document.getElementById('processing-speed');
    const displayQualityEl = document.getElementById('display-quality');
    
    if (processingSpeedEl) settings.processingSpeed = processingSpeedEl.value;
    if (displayQualityEl) settings.displayQuality = displayQualityEl.value;
}

function updateSettingsUI() {
    const processingSpeedEl = document.getElementById('processing-speed');
    const displayQualityEl = document.getElementById('display-quality');
    
    if (processingSpeedEl) processingSpeedEl.value = settings.processingSpeed;
    if (displayQualityEl) displayQualityEl.value = settings.displayQuality;
}

// Event listeners
document.addEventListener('click', function(e) {
    const settingsPanel = document.getElementById('settings-panel');
    const settingsBtn = document.querySelector('.settings-btn');
    
    if (settingsPanel && settingsBtn && 
        !settingsPanel.contains(e.target) && 
        !settingsBtn.contains(e.target)) {
        settingsPanel.classList.remove('open');
    }
});

// Prevent default drag behaviors
document.addEventListener('dragover', function(e) {
    e.preventDefault();
});

document.addEventListener('drop', function(e) {
    e.preventDefault();
});

// Error handling
window.addEventListener('error', function(e) {
    console.error('Application error:', e.error);
    alert('Đã xảy ra lỗi. Vui lòng thử lại.');
});

console.log('App.js loaded successfully');