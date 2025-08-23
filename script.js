document.addEventListener('DOMContentLoaded', () => {
    const vaeButton = document.getElementById('run-vae');
    const ganButton = document.getElementById('run-gan');
    const vaeOutput = document.getElementById('vae-output');
    const ganOutput = document.getElementById('gan-output');

    const generatedDigitsUrl = 'https://i.stack.imgur.com/yZ46j.png';

    vaeButton.addEventListener('click', () => {
        vaeOutput.innerHTML = `<p>Generating digits with VAE...</p><img src="${generatedDigitsUrl}" alt="Generated MNIST digits by VAE">`;
    });

    ganButton.addEventListener('click', () => {
        ganOutput.innerHTML = `<p>Generating digits with GAN...</p><img src="${generatedDigitsUrl}" alt="Generated MNIST digits by GAN">`;
    });
});
