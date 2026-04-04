/* Global app script (theme, nav, PWA, and loading UX) */

(() => {
    if (window.__vyaktMainInitialized) {
        return;
    }
    window.__vyaktMainInitialized = true;

    let deferredPrompt = null;

    const hideLoadingOverlay = () => {
        const overlay = document.getElementById('appLoadingOverlay');
        if (!overlay) return;
        overlay.classList.add('hidden');
    };

    const setInstallButtonVisibility = (visible) => {
        const installBtn = document.getElementById('installAppBtn');
        if (!installBtn) return;
        installBtn.classList.toggle('hidden', !visible);
    };

    const promptInstall = async () => {
        if (!deferredPrompt) {
            return;
        }

        deferredPrompt.prompt();
        const choiceResult = await deferredPrompt.userChoice;
        if (choiceResult.outcome === 'accepted') {
            console.log('User accepted the install prompt');
        } else {
            console.log('User dismissed the install prompt');
        }

        deferredPrompt = null;
        setInstallButtonVisibility(false);
    };

    const initPwaInstallFlow = () => {
        const installBtn = document.getElementById('installAppBtn');
        if (installBtn) {
            installBtn.addEventListener('click', () => {
                promptInstall().catch((err) => console.log('Install prompt error:', err));
            }, { once: false });
        }

        window.addEventListener('beforeinstallprompt', (event) => {
            event.preventDefault();
            deferredPrompt = event;
            setInstallButtonVisibility(true);
        });

        window.addEventListener('appinstalled', () => {
            deferredPrompt = null;
            setInstallButtonVisibility(false);
            console.log('PWA installed successfully');
        });
    };

    const initServiceWorker = () => {
        if (!('serviceWorker' in navigator)) {
            return;
        }

        window.addEventListener('load', () => {
            navigator.serviceWorker.getRegistration('/static/service-worker.js')
                .then((existingReg) => existingReg || navigator.serviceWorker.register('/static/service-worker.js'))
                .then(() => console.log('Service Worker registered'))
                .catch((err) => console.log('Service Worker error:', err));
        });
    };

    const initTheme = () => {
        const themeToggleBtn = document.getElementById('theme-toggle');
        const currentTheme = localStorage.getItem('theme') || 'light';

        const updateIcon = (theme) => {
            if (!themeToggleBtn) return;
            themeToggleBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
            themeToggleBtn.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
        };

        const setTheme = (theme) => {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
            updateIcon(theme);
        };

        setTheme(currentTheme);

        if (themeToggleBtn) {
            themeToggleBtn.addEventListener('click', () => {
                const activeTheme = document.documentElement.getAttribute('data-theme');
                setTheme(activeTheme === 'dark' ? 'light' : 'dark');
            });
        }
    };

    const initDropdowns = () => {
        const closeAllDropdowns = (exceptButton = null) => {
            const allDropdowns = document.querySelectorAll('.dropdown-toggle.active');
            allDropdowns.forEach((btn) => {
                if (btn === exceptButton) return;
                btn.classList.remove('active');
                const menu = btn.nextElementSibling;
                if (menu && menu.classList.contains('dropdown-menu')) {
                    menu.classList.remove('show');
                    btn.setAttribute('aria-expanded', 'false');
                }
            });
        };

        document.addEventListener('click', (event) => {
            const toggleBtn = event.target.closest('.dropdown-toggle');

            if (toggleBtn) {
                event.preventDefault();
                event.stopPropagation();

                const menu = toggleBtn.nextElementSibling;
                const isExpanded = toggleBtn.getAttribute('aria-expanded') === 'true';

                closeAllDropdowns(toggleBtn);
                if (menu && menu.classList.contains('dropdown-menu')) {
                    toggleBtn.classList.toggle('active');
                    menu.classList.toggle('show');
                    toggleBtn.setAttribute('aria-expanded', String(!isExpanded));
                }
                return;
            }

            if (!event.target.closest('.dropdown-menu')) {
                closeAllDropdowns();
            }
        });
    };

    const initMobileMenu = () => {
        const navToggle = document.querySelector('.navbar-toggler');
        const navMenu = document.querySelector('.nav-links');
        if (!navToggle || !navMenu) return;

        navToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
            const expanded = navToggle.getAttribute('aria-expanded') === 'true';
            navToggle.setAttribute('aria-expanded', String(!expanded));
        });
    };

    document.addEventListener('DOMContentLoaded', () => {
        initTheme();
        initDropdowns();
        initMobileMenu();
        initPwaInstallFlow();

        // Keep overlay visible briefly for app-like startup feel.
        window.setTimeout(hideLoadingOverlay, 400);
    });

    initServiceWorker();
})();
