/*
 * waves.c - RGB scrolling audio visualizer for Termux
 *
 * Audio input: PulseAudio-compatible server (libpulse-simple).
 * Display: ncursesw + 256-color RGB-cube approximation.
 *
 * Build:
 *   pkg install clang ncurses pulseaudio
 *   pkg install libpulse
 *   clang -O2 -Wall -Wextra waves.c -o waves \
 *       -lncursesw -lpulse-simple -lpulse -lm -pthread
 *
 * Run:
 *   ./waves
 * Optional source override:
 *   PULSE_SOURCE=your_source_name ./waves
 *
 * Keys: q quit | +/- speed | [/] wave amplitude | c cycle palette
 */

#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE_EXTENDED 1
#include <ncurses.h>
#include <pulse/simple.h>
#include <pulse/error.h>
#include <pulse/sample.h>
#include <pthread.h>
#include <math.h>
#include <stdint.h>
#include <wchar.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>

#define SAMPLE_RATE 44100
#define FFT_N 1024
#define AUDIO_CHUNK 512
#define PI 3.14159265358979323846
#define TWO_PI (2.0 * PI)
#define MAX_COLORS 216
#define COLOR_BASE 16
#define AUDIO_RING 8192

typedef struct { double r, i; } Complex;

static float audio_ring[AUDIO_RING];
static size_t ring_write = 0;
static pthread_mutex_t ring_mutex = PTHREAD_MUTEX_INITIALIZER;
static volatile int running = 1;
static volatile int audio_ok = 0;
static double speed = 1.0;
static double amplitude = 1.0;
static int palette_mode = 0;
static double bass = 0.0, mid = 0.0, treble = 0.0, overall = 0.0;
static double running_peak = 0.1;
static int palette_colors = 216;
static double phase = 0.0;

static void on_signal(int sig) { (void)sig; running = 0; }

static double clampd(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static void *audio_thread(void *unused) {
    (void)unused;
    int error = 0;
    pa_sample_spec ss = {
        .format = PA_SAMPLE_FLOAT32LE,
        .rate = SAMPLE_RATE,
        .channels = 1
    };
    const char *source = getenv("PULSE_SOURCE");
    pa_simple *pa = pa_simple_new(NULL, "Termux RGB Visualizer",
                                  PA_STREAM_RECORD,
                                  (source && *source) ? source : "OpenSL_ES_sink.monitor",
                                  "Internal audio playback input", &ss, NULL, NULL, &error);
    if (!pa) {
        fprintf(stderr, "PulseAudio capture failed: %s\n",
                pa_strerror(error));
        return NULL;
    }

    float chunk[AUDIO_CHUNK];
    audio_ok = 1;
    while (running) {
        if (pa_simple_read(pa, chunk, sizeof(chunk), &error) < 0) {
            fprintf(stderr, "PulseAudio read failed: %s\n", pa_strerror(error));
            audio_ok = 0;
            break;
        }
        pthread_mutex_lock(&ring_mutex);
        for (int i = 0; i < AUDIO_CHUNK; ++i) {
            audio_ring[ring_write] = chunk[i];
            ring_write = (ring_write + 1) % AUDIO_RING;
        }
        pthread_mutex_unlock(&ring_mutex);
    }
    pa_simple_free(pa);
    return NULL;
}

static void fft(Complex *a, int n) {
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            Complex t = a[i]; a[i] = a[j]; a[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -TWO_PI / len;
        Complex wlen = { cos(ang), sin(ang) };
        for (int i = 0; i < n; i += len) {
            Complex w = {1.0, 0.0};
            for (int k = 0; k < len / 2; ++k) {
                Complex u = a[i + k];
                Complex v = {
                    a[i + k + len/2].r*w.r - a[i + k + len/2].i*w.i,
                    a[i + k + len/2].r*w.i + a[i + k + len/2].i*w.r
                };
                a[i + k].r = u.r + v.r;
                a[i + k].i = u.i + v.i;
                a[i + k + len/2].r = u.r - v.r;
                a[i + k + len/2].i = u.i - v.i;
                double nr = w.r*wlen.r - w.i*wlen.i;
                double ni = w.r*wlen.i + w.i*wlen.r;
                w.r = nr; w.i = ni;
            }
        }
    }
}

static void analyze_audio(void) {
    float samples[FFT_N];
    pthread_mutex_lock(&ring_mutex);
    size_t p = ring_write;
    for (int i = FFT_N - 1; i >= 0; --i) {
        p = (p + AUDIO_RING - 1) % AUDIO_RING;
        samples[i] = audio_ring[p];
    }
    pthread_mutex_unlock(&ring_mutex);

    Complex a[FFT_N];
    double bsum = 0, msum = 0, tsum = 0, all = 0;
    int bn = 0, mn = 0, tn = 0;
    for (int i = 0; i < FFT_N; ++i) {
        double window = 0.5 - 0.5*cos(TWO_PI*i/(FFT_N-1));
        a[i].r = samples[i] * window;
        a[i].i = 0;
    }
    fft(a, FFT_N);
    for (int k = 1; k < FFT_N/2; ++k) {
        double hz = (double)k * SAMPLE_RATE / FFT_N;
        double mag = hypot(a[k].r, a[k].i) / (FFT_N / 2.0);
        mag = sqrt(mag);
        all += mag;
        if (hz < 250) { bsum += mag; bn++; }
        else if (hz < 2500) { msum += mag; mn++; }
        else if (hz < 11000) { tsum += mag; tn++; }
    }
    double nb = bn ? bsum / bn : 0;
    double nm = mn ? msum / mn : 0;
    double nt = tn ? tsum / tn : 0;
    double na = all / (FFT_N/2);

    /* Automatic Gain Control (AGC) peak normalization */
    double current_energy = na * 10.0;
    if (current_energy > running_peak) {
        running_peak = running_peak * 0.85 + current_energy * 0.15; /* Slower attack for smoothness */
    } else {
        running_peak = running_peak * 0.995 + current_energy * 0.005; /* Slower decay */
    }
    if (running_peak < 0.05) running_peak = 0.05;

    double agc = 0.15 / running_peak;
    if (agc > 4.0) agc = 4.0;
    if (agc < 0.2) agc = 0.2;

    /* Smoothed envelopes with heavier low-pass filtering to prevent sudden spikes */
    bass   = bass   * 0.82 + clampd(nb * 8.0 * agc, 0, 1) * 0.18;
    mid    = mid    * 0.82 + clampd(nm * 10.0 * agc, 0, 1) * 0.18;
    treble = treble * 0.82 + clampd(nt * 12.0 * agc, 0, 1) * 0.18;
    overall= overall* 0.85 + clampd(na * 8.0 * agc, 0, 1) * 0.15;
}

static short rgb_to_256(int r, int g, int b) {
    int ri = (int)lrint(clampd(r,0,255) * 5.0 / 255.0);
    int gi = (int)lrint(clampd(g,0,255) * 5.0 / 255.0);
    int bi = (int)lrint(clampd(b,0,255) * 5.0 / 255.0);
    return (short)(COLOR_BASE + 36*ri + 6*gi + bi);
}

static void setup_colors(void) {
    if (!has_colors()) return;
    start_color();
    use_default_colors();
    if (COLORS < COLOR_BASE + MAX_COLORS) palette_colors = (COLORS > 16) ? COLORS-16 : 0;
    else palette_colors = MAX_COLORS;
    /* Define RGB cube when terminal supports programmable colors. */
    if (can_change_color()) {
        int idx = COLOR_BASE;
        for (int r=0; r<6; ++r)
            for (int g=0; g<6; ++g)
                for (int b=0; b<6; ++b, ++idx) {
                    init_color((short)idx, (short)(r*200), (short)(g*200),
                               (short)(b*200));
                }
    }
    /* ncurses pair IDs are limited on some builds; use as many as available. */
    int max_pairs = COLOR_PAIRS - 1;
    if (max_pairs > 240) max_pairs = 240;
    for (int i=1; i<=max_pairs; ++i) {
        short c = (short)(COLOR_BASE + ((i-1) % MAX_COLORS));
        if (c >= COLORS) c = (short)(COLORS > 1 ? COLORS-1 : 0);
        init_pair((short)i, c, -1);
    }
}

static int color_pair_for_rgb(int r, int g, int b) {
    int max_pairs = COLOR_PAIRS - 1;
    if (max_pairs <= 0) return 0;
    int color = (int)rgb_to_256(r,g,b);
    int offset = color - COLOR_BASE;
    if (offset < 0) offset = 0;
    return 1 + (offset % (max_pairs > 240 ? 240 : max_pairs));
}

static void spectrum_color(double mix, double energy, int *r, int *g, int *b) {
    double low = clampd(bass * 1.2 + 0.1, 0, 1);
    double hue = fmod(mix * 1.4 + bass * 0.3 - treble * 0.2 + (phase * 0.05) + (palette_mode == 1 ? 0.33 : (palette_mode == 2 ? 0.66 : 0.0)), 1.0);
    if (hue < 0) hue += 1.0;

    double sector = hue * 6.0;
    int si = (int)floor(sector);
    double f = sector - si;
    double p = 0.0;
    double q = 1.0 - f;
    double t = f;
    double rr = 0, gg = 0, bb = 0;

    switch (si % 6) {
        case 0: rr = 1.0; gg = t;   bb = p;   break;
        case 1: rr = q;   gg = 1.0; bb = p;   break;
        case 2: rr = p;   gg = 1.0; bb = t;   break;
        case 3: rr = p;   gg = q;   bb = 1.0; break;
        case 4: rr = t;   gg = p;   bb = 1.0; break;
        default:rr = 1.0; gg = p;   bb = q;   break;
    }

    double boost = clampd(energy * 1.25 + 0.3 + (low * 0.2), 0.25, 1.0);
    *r = (int)clampd(rr * boost * 255.0, 0, 255);
    *g = (int)clampd(gg * boost * 255.0, 0, 255);
    *b = (int)clampd(bb * boost * 255.0, 0, 255);
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(void) {
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    pthread_t tid;
    if (pthread_create(&tid, NULL, audio_thread, NULL) != 0) {
        fprintf(stderr, "Could not start audio thread.\n");
        return 1;
    }

    setlocale(LC_ALL, "");
    initscr();
    cbreak();
    noecho();
    curs_set(0);
    keypad(stdscr, TRUE);
    nodelay(stdscr, TRUE);
    setup_colors();

    double last = now_seconds();
    int old_h = 0, old_w = 0;

    while (running) {
        int h, w;
        getmaxyx(stdscr, h, w);
        if (h != old_h || w != old_w) {
            old_h = h; old_w = w;
            clear();
        }

        int ch = getch();
        if (ch == 'q' || ch == 'Q') { running = 0; break; }
        if (ch == '+' || ch == '=') speed = clampd(speed + 0.15, 0.1, 5.0);
        if (ch == '-' || ch == '_') speed = clampd(speed - 0.15, 0.1, 5.0);
        if (ch == ']') amplitude = clampd(amplitude + 0.1, 0.1, 3.0);
        if (ch == '[') amplitude = clampd(amplitude - 0.1, 0.1, 3.0);
        if (ch == 'c' || ch == 'C') palette_mode = (palette_mode + 1) % 3;

        analyze_audio();
        double now = now_seconds();
        double dt = clampd(now - last, 0.001, 0.08);
        last = now;
        
        /* Dampened phase step to prevent sudden jerky timeline jumps */
        phase += dt * speed * (1.1 + bass * 0.8 + overall * 0.4);

        erase();
        if (h < 2 || w < 2) { refresh(); napms(20); continue; }

        for (int y=0; y<h; ++y) {
            for (int x=0; x<w; ++x) {
                double nx = (double)x / (w > 1 ? w-1 : 1);
                double ny = (double)(2*y) / (2*h > 1 ? 2*h-1 : 1);
                double scroll = nx * 8.0 - phase;
                
                /* Softer frequency scaling to keep wave curvature smooth */
                double f1 = 1.0 + bass * 1.5;
                double f2 = 2.0 + mid * 2.0;
                double f3 = 3.0 + treble * 2.5;

                double wave1 = sin(scroll * f1 + phase * 0.35);
                double wave2 = sin(scroll * f2 - phase * 0.50 + cos(nx * 2.5));
                double wave3 = sin(scroll * f3 + phase * 0.85 + ny * 1.5);
                
                double center = 0.5
                    + amplitude * (0.20 * wave1 * (0.6 + bass * 0.4)
                    + 0.12 * wave2 * (0.6 + mid * 0.3)
                    + 0.06 * wave3 * (0.6 + treble * 0.2));
                center += 0.05 * sin(phase * 0.4 + nx * 2.0) * (0.3 + overall * 0.7);
                
                double thickness = 0.06 + 0.12 * overall + 0.06 * bass;
                double distance = fabs(ny - center);
                double glow = exp(-distance / (thickness + 0.01));
                
                double ripple = 0.5 + 0.5 * sin(scroll * 1.8 + phase + ny * 5.0);
                double energy = clampd(glow * (0.6 + 0.4 * ripple)
                                       * (0.4 + overall * 1.2), 0, 1);
                double hue_mix = clampd(0.15 + 0.42 * nx + 0.2 * treble
                                        - 0.1 * bass, 0, 1);
                int r1, g1, b1, r2, g2, b2;
                spectrum_color(hue_mix, energy, &r1, &g1, &b1);

                double lower_ny = ny + 1.0 / (2.0 * h);
                double lower_dist = fabs(lower_ny - center);
                double lower_glow = exp(-lower_dist / (thickness + 0.01));
                double lower_energy = clampd(lower_glow * (0.6 + 0.4 * ripple)
                                             * (0.4 + overall * 1.2), 0, 1);
                spectrum_color(clampd(hue_mix + 0.08, 0, 1), lower_energy,
                               &r2, &g2, &b2);

                int p1 = color_pair_for_rgb(r1, g1, b1);
                int p2 = color_pair_for_rgb(r2, g2, b2);
                
                if (energy > 0.055 || lower_energy > 0.055) {
                    cchar_t cell;
                    wchar_t chars[2] = { L'▀', L'\0' };
                    (void)p2;
                    setcchar(&cell, chars, A_NORMAL, (short)p1, NULL);
                    mvadd_wch(y, x, &cell);
                } else {
                    mvaddch(y, x, ' ');
                }
            }
        }

        attrset(A_BOLD);
        if (h > 0) {
            char status[160];
            snprintf(status, sizeof(status),
                     " RGB AUDIO | %s | bass %02d%% mid %02d%% treble %02d%% | speed %.1fx amp %.1fx | q quit  +/- speed  [ ] amp  c palette ",
                     audio_ok ? "PULSE" : "NO AUDIO",
                     (int)(bass*100), (int)(mid*100), (int)(treble*100),
                     speed, amplitude);
            int maxlen = w > 1 ? w-1 : 1;
            mvaddnstr(h-1, 0, status, maxlen);
        }
        refresh();
        napms(25);
    }
    running = 0;
    pthread_join(tid, NULL);
    endwin();
    return 0;
}
