
# Implementation Plan - Job Application Modal

## User Objective
Convert the generic "Apply Now" link into a functional modal popup with a specific form structure. Form submissions should be sent to `jobs@saddl.io` with all details including file attachments.

## Proposed Strategy
1.  **Frontend**: Create a reusable Modal component in `careers.html` (and any individual job pages) containing the form.
2.  **Form Handling**: Use **FormSubmit.co**.
    -   *Why?* The site is hosted on **Cloudflare Pages**, which does not have a comprehensive built-in forms service like Netlify. FormSubmit.co is a robust, free service that handles form submissions and **file uploads** (crucial for resumes) without requiring backend code or complex API setups.
    -   It supports AJAX, allowing us to handle submissions nicely within the modal without redirecting the user away.
3.  **Styling**: Use existing Tailwind CSS classes to match the site's design (Inter font, Slate colors, Blue accents).

## Tasks
1.  **Analyze Existing Pages**: Check `careers.html` and `jobs/` folder.
2.  **Create Modal HTML**: Implement the form fields (Name, Contact, Email, Location, Experience, CTC, Notice Period, Last Working Day, Resume).
3.  **Add Interactivity**: Write JavaScript to:
    -   Open/Close modal.
    -   **Handle AJAX Submission**: Prevent default submission, POST to `https://formsubmit.co/jobs@saddl.io`, show success state, and handle errors.
4.  **Configuration**: Remove Netlify attributes. Add FormSubmit hidden fields (`_subject`, `_captcha`, etc.).

## Questions/Clarifications
-   I will assume the site is hosted on Netlify due to the presence of `netlify.toml`. If not, we'd need a third-party service like Formspree.

