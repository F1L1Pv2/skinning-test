#include "MinWin.h"
#include <mmsystem.h>
#include <stdbool.h>
#include <malloc.h>
#include <stdint.h>
#include <shellapi.h>

#include "platform.h"
#include "platform_globals.h"
#include "input.h"

void platform_fill_keycode_lookup_table();

static bool window_open = false;
static bool error = false;

bool platform_still_running(){
    if(!window_open) timeEndPeriod(1);
    return window_open;
}

static bool s_drag_and_drop_available = false;
static char** s_dropped_files = NULL;
static int s_dropped_files_count = 0;
static bool trackingMouse = false;

static bool s_mouse_locked = false;
static POINT s_center_point;

LRESULT HandleMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg){
        case WM_DROPFILES: {
            HDROP hDrop = (HDROP)wParam;
            
            // Free previous drop data if exists
            if (s_dropped_files) {
                for (int i = 0; i < s_dropped_files_count; i++) {
                    free(s_dropped_files[i]);
                }
                free(s_dropped_files);
            }

            // Get number of dropped files
            s_dropped_files_count = DragQueryFileA(hDrop, 0xFFFFFFFF, NULL, 0);
            
            if (s_dropped_files_count > 0) {
                s_dropped_files = (char**)malloc(s_dropped_files_count * sizeof(char*));
                
                for (int i = 0; i < s_dropped_files_count; i++) {
                    // Get filename length
                    UINT len = DragQueryFileA(hDrop, i, NULL, 0);
                    s_dropped_files[i] = (char*)malloc(len + 1);
                    // Get actual filename
                    DragQueryFileA(hDrop, i, s_dropped_files[i], len + 1);
                }
                
                s_drag_and_drop_available = true;
            }
            
            DragFinish(hDrop);
            break;
        }


        case WM_CLOSE: window_open = false; break;
        case WM_SIZE: {
            if(!platform_resize_window_callback(wParam == SIZE_MINIMIZED)) error = true;
            break;
        }
        case WM_SETCURSOR: {
            if (LOWORD(lParam) == HTCLIENT) {
                SetCursor(LoadCursorW(NULL, (const unsigned short *)IDC_ARROW));
                return TRUE;
            }
            break;
        }

        case WM_MOUSEMOVE:
        {
            if (s_mouse_locked) break;
            input.mouse_x = LOWORD(lParam);
            input.mouse_y = HIWORD(lParam);

            if (!trackingMouse) {
                TRACKMOUSEEVENT tme = { sizeof(TRACKMOUSEEVENT) };
                tme.dwFlags = TME_LEAVE;
                tme.hwndTrack = hwnd;
                TrackMouseEvent(&tme);
                trackingMouse = true;
            }
            break;
        }

        case WM_INPUT:
        {
            if (!s_mouse_locked) break;

            RAWINPUT raw;
            UINT size = sizeof(raw);

            GetRawInputData((HRAWINPUT)lParam, RID_INPUT, &raw, &size,
                            sizeof(RAWINPUTHEADER));

            if (raw.header.dwType == RIM_TYPEMOUSE) {

                LONG dx = raw.data.mouse.lLastX;
                LONG dy = raw.data.mouse.lLastY;

                // FPS-style accumulated deltas per frame
                input.mouse_x += dx;
                input.mouse_y += dy;

                // recenter cursor (important to avoid hitting window borders)
                SetCursorPos(s_center_point.x, s_center_point.y);
            }
            return 0;
        }

        case WM_MOUSELEAVE: {
            // Mouse has left client area
            trackingMouse = false;
            input.mouse_x = -1;
            input.mouse_y = -1;
            break;
        }

        case WM_MOUSEWHEEL:{
            int zDelta = GET_WHEEL_DELTA_WPARAM(wParam);
            input.scroll = zDelta;

            break;
        }

        case WM_KEYDOWN:
        case WM_KEYUP:
        case WM_SYSKEYDOWN:
        case WM_SYSKEYUP:
        {
            bool isDown = (msg == WM_KEYDOWN) || (msg == WM_SYSKEYDOWN) ||
                            (msg == WM_LBUTTONDOWN);

            KeyCodeID keyCode = KeyCodeLookupTable[wParam];
            Key* key = &input.keys[keyCode];

            key->isDown = isDown;
            if(key->oldIsDown == !isDown){
                if(isDown){
                    key->justPressed = 1;
                }else{
                    key->justReleased = 1;
                }
            }
            key->oldIsDown = isDown;

            break;
        }

        case WM_LBUTTONDOWN:
        case WM_RBUTTONDOWN:
        case WM_MBUTTONDOWN:
        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_MBUTTONUP:
        {
            bool isDown = (msg == WM_LBUTTONDOWN || msg == WM_RBUTTONDOWN || msg == WM_MBUTTONDOWN);
            int mouseCode = 
                (msg == WM_LBUTTONDOWN || msg == WM_LBUTTONUP)? VK_LBUTTON: 
                (msg == WM_MBUTTONDOWN || msg == WM_MBUTTONUP)? VK_MBUTTON: VK_RBUTTON;

            KeyCodeID keyCode = KeyCodeLookupTable[mouseCode];
            Key* key = &input.keys[keyCode];

            key->isDown = isDown;
            if(key->oldIsDown == !isDown){
                if(isDown){
                    key->justPressed = 1;
                }else{
                    key->justReleased = 1;
                }
            }
            key->oldIsDown = isDown;

            if(isDown){
                SetCapture(hwnd);
            }else{
                ReleaseCapture();
            }

            break;
        }
    }
    return DefWindowProcA(hWnd, msg, wParam, lParam);
}

HINSTANCE hInstance;
HWND hwnd;

void platform_create_window(const char* title, size_t width, size_t height){
    RECT wr;
    wr.left = 100;
    wr.right = width + wr.left;
    wr.top = 100;
    wr.bottom = height + wr.top;
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, false);

    hInstance = GetModuleHandleA(NULL);

    WNDCLASSEXA windowClass = {0};
    windowClass.cbSize = sizeof(WNDCLASSEXA);
    windowClass.style = 0;
    windowClass.lpfnWndProc = HandleMsg;
    windowClass.cbClsExtra = 0;
    windowClass.cbWndExtra = 0;
    windowClass.hInstance = hInstance;
    windowClass.hIcon = NULL;
    windowClass.hCursor = NULL;
    windowClass.hbrBackground = NULL;
    windowClass.lpszMenuName = "TRIEX";
    windowClass.lpszClassName = "TRIEX";
    windowClass.hIconSm = NULL;
    RegisterClassExA(&windowClass);

    hwnd = CreateWindowExA(
        0,
        "TRIEX",
        title,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, wr.right - wr.left, wr.bottom - wr.top,
        NULL, NULL, hInstance, NULL
    );

    ShowWindow(hwnd, SW_SHOW);
    DragAcceptFiles(hwnd, TRUE);
    window_open = true;

    platform_fill_keycode_lookup_table();

    timeBeginPeriod(1);

    RAWINPUTDEVICE rid;
    rid.usUsagePage = 0x01;  // Generic Desktop Controls
    rid.usUsage = 0x02;      // Mouse
    rid.dwFlags = RIDEV_INPUTSINK;
    rid.hwndTarget = hwnd;

    RegisterRawInputDevices(&rid, 1, sizeof(rid));
}

MSG msg;

bool platform_window_handle_events(){
    input.scroll = 0;

    if (s_mouse_locked) {
        input.mouse_x = 0;
        input.mouse_y = 0;
    }
    
    for(int i = 0; i < sizeof(input.keys) / sizeof(input.keys[0]); i++){
        input.keys[i].justPressed = 0;
        input.keys[i].justReleased = 0;
    }

    while(PeekMessageA(&msg,NULL,0,0,PM_REMOVE)){
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
        if(error) break;
    }
    return !error;
}

static void busy_wait_ticks(long long ticks_to_wait)
{
    LARGE_INTEGER start, cur;
    QueryPerformanceCounter(&start);
    long long target = start.QuadPart + ticks_to_wait;

    for (;;) {
        QueryPerformanceCounter(&cur);
        if (cur.QuadPart >= target)
            break;
        YieldProcessor();   // pause instruction / CPU hint
    }
}


void platform_sleep(size_t milis)
{
    if (milis == 0)
        return;

    const double spin_threshold_ms = 1.5; // tune: 1–3 ms is typical

    LARGE_INTEGER freq_li;
    QueryPerformanceFrequency(&freq_li);
    long long freq = freq_li.QuadPart;

    double ticks_per_ms = (double)freq / 1000.0;
    long long total_ticks = (long long)( (double)milis * ticks_per_ms + 0.5 );

    if ((double)milis > spin_threshold_ms)
    {
        // Sleep for the coarse part
        size_t coarse_ms = (size_t)((double)milis - spin_threshold_ms);
        if (coarse_ms > 0)
            Sleep((DWORD) coarse_ms);

        // Now busy-spin the remainder
        long long spin_ticks =
            (long long)(spin_threshold_ms * ticks_per_ms + 0.5);

        busy_wait_ticks(spin_ticks);
    }
    else
    {
        // Entire wait fits inside high-res busy loop
        busy_wait_ticks(total_ticks);
    }
}

uint64_t platform_get_time_milis(){
    FILETIME ft;
    ULARGE_INTEGER uli;
    GetSystemTimeAsFileTime(&ft);
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    return (uli.QuadPart - 116444736000000000ULL) / 10000ULL;
}

uint64_t platform_get_time_nanos() {
    FILETIME ft;
    ULARGE_INTEGER uli;
    GetSystemTimeAsFileTime(&ft);
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    return (uli.QuadPart - 116444736000000000ULL) * 100ULL;
}

void platform_fill_keycode_lookup_table(){
  KeyCodeLookupTable[VK_LBUTTON] = KEY_MOUSE_LEFT;
  KeyCodeLookupTable[VK_MBUTTON] = KEY_MOUSE_MIDDLE;
  KeyCodeLookupTable[VK_RBUTTON] = KEY_MOUSE_RIGHT;
  
  KeyCodeLookupTable['A'] = KEY_A;
  KeyCodeLookupTable['B'] = KEY_B;
  KeyCodeLookupTable['C'] = KEY_C;
  KeyCodeLookupTable['D'] = KEY_D;
  KeyCodeLookupTable['E'] = KEY_E;
  KeyCodeLookupTable['F'] = KEY_F;
  KeyCodeLookupTable['G'] = KEY_G;
  KeyCodeLookupTable['H'] = KEY_H;
  KeyCodeLookupTable['I'] = KEY_I;
  KeyCodeLookupTable['J'] = KEY_J;
  KeyCodeLookupTable['K'] = KEY_K;
  KeyCodeLookupTable['L'] = KEY_L;
  KeyCodeLookupTable['M'] = KEY_M;
  KeyCodeLookupTable['N'] = KEY_N;
  KeyCodeLookupTable['O'] = KEY_O;
  KeyCodeLookupTable['P'] = KEY_P;
  KeyCodeLookupTable['Q'] = KEY_Q;
  KeyCodeLookupTable['R'] = KEY_R;
  KeyCodeLookupTable['S'] = KEY_S;
  KeyCodeLookupTable['T'] = KEY_T;
  KeyCodeLookupTable['U'] = KEY_U;
  KeyCodeLookupTable['V'] = KEY_V;
  KeyCodeLookupTable['W'] = KEY_W;
  KeyCodeLookupTable['X'] = KEY_X;
  KeyCodeLookupTable['Y'] = KEY_Y;
  KeyCodeLookupTable['Z'] = KEY_Z;
  KeyCodeLookupTable['0'] = KEY_0;
  KeyCodeLookupTable['1'] = KEY_1;
  KeyCodeLookupTable['2'] = KEY_2;
  KeyCodeLookupTable['3'] = KEY_3;
  KeyCodeLookupTable['4'] = KEY_4;
  KeyCodeLookupTable['5'] = KEY_5;
  KeyCodeLookupTable['6'] = KEY_6;
  KeyCodeLookupTable['7'] = KEY_7;
  KeyCodeLookupTable['8'] = KEY_8;
  KeyCodeLookupTable['9'] = KEY_9;
  
  KeyCodeLookupTable[VK_SPACE] = KEY_SPACE,
  KeyCodeLookupTable[VK_OEM_3] = KEY_TICK,
  KeyCodeLookupTable[VK_OEM_MINUS] = KEY_MINUS,

  KeyCodeLookupTable[VK_OEM_PLUS] = KEY_EQUAL,
  KeyCodeLookupTable[VK_OEM_4] = KEY_LEFT_BRACKET,
  KeyCodeLookupTable[VK_OEM_6] = KEY_RIGHT_BRACKET,
  KeyCodeLookupTable[VK_OEM_1] = KEY_SEMICOLON,
  KeyCodeLookupTable[VK_OEM_7] = KEY_QUOTE,
  KeyCodeLookupTable[VK_OEM_COMMA] = KEY_COMMA,
  KeyCodeLookupTable[VK_OEM_PERIOD] = KEY_PERIOD,
  KeyCodeLookupTable[VK_OEM_2] = KEY_FORWARD_SLASH,
  KeyCodeLookupTable[VK_OEM_5] = KEY_BACKWARD_SLASH,
  KeyCodeLookupTable[VK_TAB] = KEY_TAB,
  KeyCodeLookupTable[VK_ESCAPE] = KEY_ESCAPE,
  KeyCodeLookupTable[VK_PAUSE] = KEY_PAUSE,
  KeyCodeLookupTable[VK_UP] = KEY_UP,
  KeyCodeLookupTable[VK_DOWN] = KEY_DOWN,
  KeyCodeLookupTable[VK_LEFT] = KEY_LEFT,
  KeyCodeLookupTable[VK_RIGHT] = KEY_RIGHT,
  KeyCodeLookupTable[VK_BACK] = KEY_BACKSPACE,
  KeyCodeLookupTable[VK_RETURN] = KEY_RETURN,
  KeyCodeLookupTable[VK_DELETE] = KEY_DELETE,
  KeyCodeLookupTable[VK_INSERT] = KEY_INSERT,
  KeyCodeLookupTable[VK_HOME] = KEY_HOME,
  KeyCodeLookupTable[VK_END] = KEY_END,
  KeyCodeLookupTable[VK_PRIOR] = KEY_PAGE_UP,
  KeyCodeLookupTable[VK_NEXT] = KEY_PAGE_DOWN,
  KeyCodeLookupTable[VK_CAPITAL] = KEY_CAPS_LOCK,
  KeyCodeLookupTable[VK_NUMLOCK] = KEY_NUM_LOCK,
  KeyCodeLookupTable[VK_SCROLL] = KEY_SCROLL_LOCK,
  KeyCodeLookupTable[VK_APPS] = KEY_MENU,
  
  KeyCodeLookupTable[VK_SHIFT] = KEY_SHIFT,
  KeyCodeLookupTable[VK_LSHIFT] = KEY_SHIFT,
  KeyCodeLookupTable[VK_RSHIFT] = KEY_SHIFT,
  
  KeyCodeLookupTable[VK_CONTROL] = KEY_CONTROL,
  KeyCodeLookupTable[VK_LCONTROL] = KEY_CONTROL,
  KeyCodeLookupTable[VK_RCONTROL] = KEY_CONTROL,
  
  KeyCodeLookupTable[VK_MENU] = KEY_ALT,
  KeyCodeLookupTable[VK_LMENU] = KEY_ALT,
  KeyCodeLookupTable[VK_RMENU] = KEY_ALT,
  
  KeyCodeLookupTable[VK_F1] = KEY_F1;
  KeyCodeLookupTable[VK_F2] = KEY_F2;
  KeyCodeLookupTable[VK_F3] = KEY_F3;
  KeyCodeLookupTable[VK_F4] = KEY_F4;
  KeyCodeLookupTable[VK_F5] = KEY_F5;
  KeyCodeLookupTable[VK_F6] = KEY_F6;
  KeyCodeLookupTable[VK_F7] = KEY_F7;
  KeyCodeLookupTable[VK_F8] = KEY_F8;
  KeyCodeLookupTable[VK_F9] = KEY_F9;
  KeyCodeLookupTable[VK_F10] = KEY_F10;
  KeyCodeLookupTable[VK_F11] = KEY_F11;
  KeyCodeLookupTable[VK_F12] = KEY_F12;
  
  KeyCodeLookupTable[VK_NUMPAD0] = KEY_NUMPAD_0;
  KeyCodeLookupTable[VK_NUMPAD1] = KEY_NUMPAD_1;
  KeyCodeLookupTable[VK_NUMPAD2] = KEY_NUMPAD_2;
  KeyCodeLookupTable[VK_NUMPAD3] = KEY_NUMPAD_3;
  KeyCodeLookupTable[VK_NUMPAD4] = KEY_NUMPAD_4;
  KeyCodeLookupTable[VK_NUMPAD5] = KEY_NUMPAD_5;
  KeyCodeLookupTable[VK_NUMPAD6] = KEY_NUMPAD_6;
  KeyCodeLookupTable[VK_NUMPAD7] = KEY_NUMPAD_7;
  KeyCodeLookupTable[VK_NUMPAD8] = KEY_NUMPAD_8;
  KeyCodeLookupTable[VK_NUMPAD9] = KEY_NUMPAD_9;
}

void platform_set_mouse_position(size_t x, size_t y) {
    POINT pt = { (LONG)x, (LONG)y };
    ClientToScreen(hwnd, &pt); // Convert to screen coordinates
    SetCursorPos(pt.x, pt.y);
}

uint64_t pre_fullScreenSizeX;
uint64_t pre_fullScreenSizeY;
uint64_t pre_fullScreenPosX;
uint64_t pre_fullScreenPosY;

void platform_enable_fullscreen(){
    RECT rect;
    if(GetWindowRect(hwnd, &rect)){
        pre_fullScreenSizeX = rect.right - rect.left;
        pre_fullScreenSizeY = rect.bottom - rect.top;
        pre_fullScreenPosX = rect.left;
        pre_fullScreenPosY = rect.top;
    }
    
    //disable window decoration
    LONG style = GetWindowLong(hwnd, GWL_STYLE);
    style &= ~(WS_OVERLAPPEDWINDOW);
    SetWindowLong(hwnd, GWL_STYLE, style);

    uint64_t posX;
    uint64_t posY;
    uint64_t sizeX;
    uint64_t sizeY;
    
    HMONITOR hMonitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
    if (hMonitor == NULL) {
        return;
    }
    MONITORINFO monitorInfo;
    monitorInfo.cbSize = sizeof(MONITORINFO);
    if (GetMonitorInfo(hMonitor, &monitorInfo)) {
        posX = monitorInfo.rcMonitor.left;
        posY = monitorInfo.rcMonitor.top;
        sizeX = monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left;
        sizeY = monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top;
    } else {
       return;
    }

    SetWindowPos(hwnd, NULL, posX, posY, sizeX, sizeY, SWP_FRAMECHANGED);
}

void platform_disable_fullscreen(){
    //enable window decoration
    LONG style = GetWindowLong(hwnd, GWL_STYLE);
    style |= (WS_OVERLAPPEDWINDOW);
    SetWindowLong(hwnd, GWL_STYLE, style);

    SetWindowPos(hwnd, NULL, pre_fullScreenPosX, pre_fullScreenPosY, pre_fullScreenSizeX, pre_fullScreenSizeY, SWP_FRAMECHANGED);
}

bool platform_drag_and_drop_available() {
    return s_drag_and_drop_available;
}

char** platform_get_drag_and_drop_files(int* count) {
    *count = s_dropped_files_count;
    char** result = (char**)s_dropped_files;
    
    s_dropped_files = NULL;
    s_dropped_files_count = 0;
    s_drag_and_drop_available = false;
    
    return result;
}

void platform_release_drag_and_drop(char** files, int count){
    if(files == NULL) return;

    for(int i = 0; i < count; i++){
        if(files[i] != NULL) free(files[i]);
    }

    free(files);
}

bool platform_free_dynamic_library(void* dll)
{
  BOOL freeResult = FreeLibrary((HMODULE)dll);

  return (bool)freeResult;
}

void* platform_load_dynamic_library(const char* dll)
{
  HMODULE result = LoadLibraryA(dll);

  return result;
}

void* platform_load_dynamic_function(void* dll, const char* funName)
{
  FARPROC proc = GetProcAddress((HMODULE)dll, funName);

  return (void*)proc;
}

void platform_lock_mouse()
{
    if (s_mouse_locked) return;

    s_mouse_locked = true;

    // Hide cursor
    ShowCursor(FALSE);

    // Center point in screen coords
    RECT rect;
    GetClientRect(hwnd, &rect);

    POINT center = {
        (rect.right - rect.left) / 2,
        (rect.bottom - rect.top) / 2
    };

    ClientToScreen(hwnd, &center);
    s_center_point = center;

    // Move cursor to center
    SetCursorPos(center.x, center.y);

    // Lock cursor to window
    RECT clip = { rect.left, rect.top, rect.right, rect.bottom };
    ClientToScreen(hwnd, (POINT*)&clip.left);
    ClientToScreen(hwnd, (POINT*)&clip.right);
    ClipCursor(&clip);
}

void platform_unlock_mouse()
{
    if (!s_mouse_locked) return;

    s_mouse_locked = false;

    // Show cursor
    ShowCursor(TRUE);

    // Release lock
    ClipCursor(NULL);
}

#ifndef DEBUG

int main();

int WinMain(
  HINSTANCE hInstance,
  HINSTANCE hPrevInstance,
  LPSTR     lpCmdLine,
  int       nShowCmd
){
    return main();
}

#endif
