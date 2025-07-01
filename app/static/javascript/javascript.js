const sections = document.querySelectorAll('.content-section');
const menuLinks = document.querySelectorAll('#menu .nav-link');

function showSection(id) {
    sections.forEach(sec => sec.classList.remove('active'));
    document.getElementById(id).classList.add('active');
}

menuLinks.forEach(link => {
    link.addEventListener('click', e => {
        e.preventDefault();
        menuLinks.forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        showSection(link.dataset.target);
    });
});

// Mostrar solo INICIO al cargar
window.addEventListener('DOMContentLoaded', () => {
    showSection('inicio');
    document.querySelector('#menu .nav-link[data-target="inicio"]').classList.add('active');
});
