const motionButton = document.querySelector('.motion-toggle');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let paused = false;

function updateMotion() {
  document.documentElement.classList.toggle('effects-paused', paused || reducedMotion.matches);
  motionButton.hidden = reducedMotion.matches;
  motionButton.setAttribute('aria-pressed', String(paused));
  motionButton.textContent = paused ? 'Resume light effects' : 'Pause light effects';
}

motionButton.addEventListener('click', () => {
  paused = !paused;
  updateMotion();
});
reducedMotion.addEventListener('change', updateMotion);
updateMotion();
