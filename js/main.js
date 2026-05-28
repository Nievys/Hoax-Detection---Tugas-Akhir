// ── Init ──────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  await go('dashboard');
  
  const qtInput = document.getElementById('qt-input');
  if (qtInput) {
    qtInput.value = 'Lo tau ga, org2 gitu emang brengsek bgt! Blm tobat juga. #kebencian https://spam.com @akun123';
  }
});
