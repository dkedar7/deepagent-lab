import { expect, test } from '@jupyterlab/galata';

/**
 * Visual-regression for the chat sidebar (pixel-diff vs committed baselines).
 *
 * Catches the recurring styling/layout inconsistencies in the extension's UI.
 * We snapshot the `.deepagents-chat-container` (the extension's own surface,
 * not JupyterLab chrome) in its ready + conversation states, masking the
 * per-message timestamps (the only dynamic content). Runs against the
 * model-free stub agent, so it's deterministic and needs no API key.
 *
 * Captured in both light + dark JupyterLab themes — the sidebar consumes the
 * `--jp-*` theme variables, so it must track the active theme.
 *
 * Baselines are generated and checked in the same CI environment (Galata
 * bundles its own Playwright, so an external pinned image would mismatch
 * browser versions). Regenerate via the ui-tests workflow (it uploads the
 * generated snapshots as an artifact to commit).
 */
const shot = { animations: 'disabled', maxDiffPixelRatio: 0.01 } as const;

async function openHealthyChat(page: any) {
  await page.waitForCondition(
    async () =>
      await page.evaluate(() =>
        Boolean(
          (window as any).jupyterapp?.commands?.hasCommand('deepagents:open-chat')
        )
      )
  );
  await page.evaluate(() =>
    (window as any).jupyterapp.commands.execute('deepagents:open-chat')
  );
  await expect(page.locator('.deepagents-chat-container')).toBeVisible();
  await expect(page.locator('.deepagents-status-healthy')).toBeVisible({
    timeout: 30_000
  });
}

for (const theme of ['light', 'dark'] as const) {
  test(`chat sidebar — ${theme}`, async ({ page }) => {
    if (theme === 'dark') {
      await page.theme.setDarkTheme();
    } else {
      await page.theme.setLightTheme();
    }

    await openHealthyChat(page);
    const chat = page.locator('.deepagents-chat-container');
    const masks = { ...shot, mask: [page.locator('.deepagents-message-time')] };

    // Ready state (header + status + "agent ready" system message).
    await expect(chat).toHaveScreenshot(`sidebar-ready-${theme}.png`, masks);

    // Conversation state (user + assistant bubble styling).
    await page.locator('.deepagents-chat-input').fill('ping');
    await page.locator('.deepagents-send-button').click();
    await expect(
      page
        .locator('.deepagents-message-assistant')
        .filter({ hasText: 'stub reply: ping' })
    ).toBeVisible({ timeout: 30_000 });
    await expect(chat).toHaveScreenshot(`sidebar-conversation-${theme}.png`, masks);
  });
}
