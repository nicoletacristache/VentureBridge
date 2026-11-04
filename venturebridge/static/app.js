(() => {
  'use strict';

  document.documentElement.classList.add('js');

  const navToggle = document.querySelector('[data-nav-toggle]');
  const nav = document.querySelector('[data-nav]');
  if (navToggle && nav) {
    navToggle.addEventListener('click', () => {
      const open = navToggle.getAttribute('aria-expanded') === 'true';
      navToggle.setAttribute('aria-expanded', String(!open));
      nav.classList.toggle('is-open', !open);
    });
  }

  document.querySelectorAll('[data-dismiss-parent]').forEach((button) => {
    button.addEventListener('click', () => {
      if (button.parentElement) button.parentElement.remove();
    });
  });

  document.querySelectorAll('[data-toggle-secret]').forEach((button) => {
    button.addEventListener('click', () => {
      const input = document.getElementById(button.dataset.target || '');
      if (!input) return;
      const reveal = input.type === 'password';
      input.type = reveal ? 'text' : 'password';
      button.textContent = reveal ? '隐藏' : '显示';
      button.setAttribute('aria-pressed', String(reveal));
    });
  });

  document.querySelectorAll('[data-toggle-region]').forEach((control) => {
    const sync = () => {
      const region = document.getElementById(control.dataset.toggleRegion || '');
      if (region) region.hidden = !control.checked;
    };
    control.addEventListener('change', sync);
    sync();
  });

  document.querySelectorAll('textarea[data-char-count], input[data-char-count]').forEach((field) => {
    const counterId = (field.getAttribute('aria-describedby') || '').split(' ').find((id) => id.endsWith('-count'));
    const counter = counterId ? document.getElementById(counterId) : null;
    const sync = () => {
      if (!counter) return;
      const maximum = field.maxLength > 0 ? ` / ${field.maxLength}` : '';
      counter.textContent = `${field.value.length}${maximum} 字`;
    };
    field.addEventListener('input', sync);
    sync();
  });

  document.querySelectorAll('[data-custom-direction]').forEach((field) => {
    const form = field.closest('form');
    if (!form) return;
    const customChoice = form.querySelector('input[name="direction_id"][value="__custom__"]');
    const choices = form.querySelectorAll('input[name="direction_id"]');
    if (!customChoice) return;
    const sync = () => {
      field.required = customChoice.checked;
      const container = field.closest('.custom-direction-field');
      if (container) container.classList.toggle('is-required', customChoice.checked);
    };
    choices.forEach((choice) => choice.addEventListener('change', sync));
    sync();
  });

  document.querySelectorAll('form[data-profile-form]').forEach((form) => {
    const profileSignals = Array.from(form.querySelectorAll('[data-profile-signal]'));
    if (!profileSignals.length) return;
    const message = '请至少填写学习目标、经历、技能、兴趣或可用资源中的一项。';
    const sync = () => {
      const hasSignal = profileSignals.some((field) => field.value.trim().length > 0);
      profileSignals.forEach((field) => field.setCustomValidity(''));
      if (!hasSignal) profileSignals[0].setCustomValidity(message);
      return hasSignal;
    };
    profileSignals.forEach((field) => field.addEventListener('input', sync));
    form.addEventListener('submit', sync);
    sync();
  });

  document.querySelectorAll('form').forEach((form) => {
    form.addEventListener('submit', (event) => {
      const submitter = event.submitter;
      if (!submitter) return;
      const message = submitter.dataset.confirmMessage;
      if (message && !window.confirm(message)) {
        event.preventDefault();
        return;
      }
      const pendingLabel = submitter.dataset.submitLabel;
      if (pendingLabel && form.checkValidity()) {
        submitter.dataset.originalLabel = submitter.textContent;
        submitter.textContent = pendingLabel;
        submitter.setAttribute('aria-disabled', 'true');
      }
    });
  });

  document.querySelectorAll('[data-copy-value]').forEach((button) => {
    button.addEventListener('click', async () => {
      const value = button.dataset.copyValue || '';
      try {
        await navigator.clipboard.writeText(value);
        button.textContent = '已复制';
      } catch (_error) {
        button.textContent = '请手动复制';
      }
    });
  });
})();
